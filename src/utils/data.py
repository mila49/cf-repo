"""
Data Handling Utilities

This module provides utilities for:
- Train/validation splitting
- Data loader creation
- Configuration loading
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader, random_split


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split
    val_ratio : float, default=0.2
        Fraction of data to use for validation (0.0 to 1.0)
    seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    train_dataset : Dataset
        Training subset
    val_dataset : Dataset
        Validation subset
        
    Raises
    ------
    ValueError
        If validation set would be empty or too small
    """
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    if val_size < 1:
        raise ValueError(
            f"Validation set is empty. Dataset size: {dataset_size}, "
            f"val_ratio: {val_ratio}"
        )
    
    if val_size < 3:
        raise ValueError(
            f"Validation set must contain at least 3 samples. "
            f"Current size: {val_size}. Consider decreasing val_ratio."
        )
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
    drop_last_train: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and validation.
    
    Parameters
    ----------
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset
    batch_size : int
        Batch size for both loaders
    num_workers : int, default=0
        Number of worker processes for data loading
    shuffle_train : bool, default=True
        Whether to shuffle training data
    drop_last_train : bool, optional
        Whether to drop last incomplete batch in training
        If None, automatically drops if train_size % batch_size == 1
        (to avoid BatchNorm issues with single-sample batches)
        
    Returns
    -------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    """
    # Auto-detect if we should drop last batch
    if drop_last_train is None:
        drop_last_train = (len(train_dataset) % batch_size == 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=num_workers,
        drop_last=False  # Always keep all validation samples
    )
    
    return train_loader, val_loader


def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load YAML configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
        Can be relative or absolute
        
    Returns
    -------
    config : dict
        Parsed configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    yaml.YAMLError
        If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict, output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    output_path : str or Path
        Output path for YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False