"""Utility functions for the Activation Maximization XAI project."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        preferred: Preferred device type ('cuda', 'mps', 'cpu').
        
    Returns:
        torch.device: The selected device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif preferred == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path.
        
    Returns:
        Path: The directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        path: Save path.
        metadata: Additional metadata to save.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Union[str, Path]
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer (optional).
        path: Checkpoint path.
        
    Returns:
        Dict containing checkpoint data.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {path}")
    return checkpoint


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially restore weights for.
            
        Returns:
            bool: True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save current model weights."""
        self.best_weights = model.state_dict().copy()


def normalize_image(image: torch.Tensor, mean: Tuple[float, ...] = (0.5,), std: Tuple[float, ...] = (0.5,)) -> torch.Tensor:
    """Normalize image tensor.
    
    Args:
        image: Input image tensor.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        
    Returns:
        torch.Tensor: Normalized image.
    """
    if image.dim() == 3:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    elif image.dim() == 4:
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
    
    return (image - mean) / std


def denormalize_image(image: torch.Tensor, mean: Tuple[float, ...] = (0.5,), std: Tuple[float, ...] = (0.5,)) -> torch.Tensor:
    """Denormalize image tensor.
    
    Args:
        image: Normalized image tensor.
        mean: Mean values used for normalization.
        std: Standard deviation values used for normalization.
        
    Returns:
        torch.Tensor: Denormalized image.
    """
    if image.dim() == 3:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    elif image.dim() == 4:
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
    
    return image * std + mean


def clamp_image(image: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """Clamp image values to specified range.
    
    Args:
        image: Input image tensor.
        min_val: Minimum value.
        max_val: Maximum value.
        
    Returns:
        torch.Tensor: Clamped image.
    """
    return torch.clamp(image, min_val, max_val)
