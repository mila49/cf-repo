from abc import abstractmethod
from pathlib import Path
import yaml
import os
import wandb
from dotenv import load_dotenv
from typing import Callable, Optional


class Pipeline:
    """
    Base class for all pipelines. Handles configuration loading and provides a
    common interface for running the pipeline, including support for Weights & Biases
    hyperparameter sweeps using grid search.
    """
    def __init__(self, config_file: str):
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir: str = "config"
        self.config = self.load_config(self.root_dir / self.config_dir / config_file)
        
        # Initialize W&B sweep configuration
        self.wb_sweep: bool = self.config.get("wb_sweep", False)
        self.sweep_id: Optional[str] = None
        self.sweep_config: dict = self._setup_sweep_config() if self.wb_sweep else None
    

    def load_config(self, config_path: Path) -> dict:
        """
        Load the configuration file.

        Args:
            config_path: The path to the configuration file.

        Returns:
            dict: The loaded configuration.
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    

    def _setup_sweep_config(self) -> Optional[dict]:
        """
        Setup W&B sweep configuration from the config file.
        
        Expected config structure:
        ```yaml
        wb_sweep: true
        parameters:
            learning_rate:
                - 0.001
                - 0.01
            latent_dim:
                - 32
                - 64
        ```
        
        This method transforms the config format into W&B's expected format with 'values' key.

        Returns:
            dict: Sweep configuration for W&B, or None if sweep is disabled.
        """
        if not self.wb_sweep:
            return None
        
        if "parameters" not in self.config:
            raise ValueError(
                "wb_sweep is enabled but 'parameters' key not found in config. "
                "Please define parameter values to sweep over."
            )
        
        # Transform parameters from list format to W&B format with 'values' key
        wb_parameters = {}
        for param_name, param_values in self.config["parameters"].items():
            if isinstance(param_values, list):
                wb_parameters[param_name] = {"values": param_values}
            else:
                raise ValueError(
                    f"Parameter '{param_name}' must be a list of values, "
                    f"got {type(param_values).__name__}"
                )
        
        sweep_config = {
            "method": "grid",
            "parameters": wb_parameters
        }
        
        return sweep_config
    

    def initialize_sweep(self, project: str, entity: str) -> None:
        """
        Initialize a W&B sweep.
        
        Args:
            project: W&B project name.
            entity: W&B entity name.
        
        Returns:
            str: The sweep ID.
        """
        if not self.wb_sweep or self.sweep_config is None:
            return
        
        load_dotenv()
        self.sweep_id = wandb.sweep(
            self.sweep_config,
            project=project,
            entity=entity,
        )
    

    def run_with_sweep(
        self, 
        train_function: Callable, 
        project: str, 
        entity: str,
        count: Optional[int] = None,
    ) -> None:
        """
        Run the pipeline with W&B sweep support.
        
        Args:
            train_function: The function to execute for each sweep configuration.
                           Will be called with no arguments. It should use wandb.config
                           to access the current hyperparameters.
            project: W&B project name.
            entity: W&B entity name.
            count: Maximum number of runs. If None, runs all configurations (for grid search).
        """
        if not self.wb_sweep:
            # If sweep is not enabled, just run normally
            train_function()
            return
        
        self.initialize_sweep(project, entity)
        
        def sweep_agent():
            with wandb.init() as run:
                # Update config with current sweep parameters
                self._update_config_from_sweep()
                train_function()
        
        wandb.agent(
            self.sweep_id,
            function=sweep_agent,
            count=count,
        )
    

    def _update_config_from_sweep(self) -> None:
        """
        Update the pipeline config with values from the current W&B sweep run.
        Called within a wandb.init() context.
        Only updates parameters that are defined in the sweep configuration.
        """
        if not self.wb_sweep or "parameters" not in self.config:
            return
        
        # Only update parameters that are part of the sweep
        sweep_param_names = self.config["parameters"].keys()
        
        for param_name in sweep_param_names:
            if param_name in wandb.config:
                self.config[param_name] = wandb.config[param_name]
    

    @abstractmethod
    def run(self):
        """
        Run the pipeline. Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in subclass")