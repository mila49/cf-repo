from abc import abstractmethod
from pathlib import Path
import yaml


class Pipeline:
    """
    Base class for all pipelines. Handles configuration loading and provides a
    common interface for running the pipeline.
    """
    def __init__(self, config_file: str):
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir: str = "config"
        self.config = self.load_config(self.root_dir / self.config_dir / config_file)
    

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
    

    @abstractmethod
    def run(self):
        """
        Run the pipeline. Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in subclass")