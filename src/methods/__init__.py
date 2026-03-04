"""Activation maximization methods for neural network interpretability."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RegularizationFunction(ABC):
    """Abstract base class for regularization functions."""
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss.
        
        Args:
            x: Input tensor.
            
        Returns:
            Regularization loss.
        """
        pass


class TotalVariationRegularization(RegularizationFunction):
    """Total variation regularization to encourage smooth images."""
    
    def __init__(self, weight: float = 0.1):
        """Initialize TV regularization.
        
        Args:
            weight: Regularization weight.
        """
        self.weight = weight
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute TV regularization loss."""
        batch_size = x.size(0)
        
        # Compute horizontal differences
        tv_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        
        # Compute vertical differences
        tv_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        
        return self.weight * (tv_h + tv_v) / batch_size


class L2Regularization(RegularizationFunction):
    """L2 regularization to prevent extreme values."""
    
    def __init__(self, weight: float = 0.01):
        """Initialize L2 regularization.
        
        Args:
            weight: Regularization weight.
        """
        self.weight = weight
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2 regularization loss."""
        return self.weight * torch.norm(x, p=2) ** 2


class BlurRegularization(RegularizationFunction):
    """Blur regularization to encourage natural-looking images."""
    
    def __init__(self, weight: float = 0.0, kernel_size: int = 3):
        """Initialize blur regularization.
        
        Args:
            weight: Regularization weight.
            kernel_size: Size of blur kernel.
        """
        self.weight = weight
        self.kernel_size = kernel_size
        
        # Create Gaussian blur kernel
        sigma = kernel_size / 6.0
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('blur_kernel', kernel)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian blur kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords = coords - size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.view(1, 1, size, size)
        
        return kernel
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute blur regularization loss."""
        if self.weight == 0:
            return torch.tensor(0.0, device=x.device)
        
        # Apply blur
        blurred = F.conv2d(x, self.blur_kernel, padding=self.kernel_size//2)
        
        # Compute difference
        return self.weight * F.mse_loss(x, blurred)


class ActivationMaximizer:
    """Activation maximization for neural network interpretability."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        target_class: Optional[int] = None,
        regularization_functions: Optional[List[RegularizationFunction]] = None,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize activation maximizer.
        
        Args:
            model: Neural network model.
            target_layer: Name of layer to maximize activation for.
            target_class: Target class for maximization (None for any class).
            regularization_functions: List of regularization functions.
            device: Device to run on.
        """
        self.model = model.to(device)
        self.target_layer = target_layer
        self.target_class = target_class
        self.device = device
        
        # Set regularization functions
        self.regularization_functions = regularization_functions or []
        
        # Ensure model is in eval mode
        self.model.eval()
        
        logger.info(f"Initialized ActivationMaximizer for layer '{target_layer}'")
    
    def maximize_activation(
        self,
        input_shape: Tuple[int, ...],
        iterations: int = 200,
        learning_rate: float = 0.01,
        optimization_method: str = "adam",
        gradient_clipping: float = 1.0,
        noise_std: float = 0.0,
        diversity_weight: float = 0.0,
        verbose: bool = True
    ) -> torch.Tensor:
        """Maximize activation for the target layer.
        
        Args:
            input_shape: Shape of input tensor (excluding batch dimension).
            iterations: Number of optimization iterations.
            learning_rate: Learning rate for optimization.
            optimization_method: Optimization method ('adam', 'sgd', 'rmsprop').
            gradient_clipping: Gradient clipping threshold.
            noise_std: Standard deviation of noise to add.
            diversity_weight: Weight for diversity regularization.
            verbose: Whether to print progress.
            
        Returns:
            Optimized input tensor.
        """
        # Initialize input
        x = torch.randn(1, *input_shape, device=self.device, requires_grad=True)
        
        # Initialize optimizer
        if optimization_method.lower() == "adam":
            optimizer = torch.optim.Adam([x], lr=learning_rate)
        elif optimization_method.lower() == "sgd":
            optimizer = torch.optim.SGD([x], lr=learning_rate)
        elif optimization_method.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop([x], lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Add noise for diversity
            if noise_std > 0:
                noise = torch.randn_like(x) * noise_std
                x_noisy = x + noise
            else:
                x_noisy = x
            
            # Get layer output
            layer_output = self.model.get_layer_output(x_noisy, self.target_layer)
            
            # Compute activation loss
            if self.target_class is not None:
                # Maximize specific class activation
                activation_loss = -layer_output[:, self.target_class].mean()
            else:
                # Maximize any activation
                activation_loss = -layer_output.mean()
            
            # Compute regularization losses
            reg_loss = torch.tensor(0.0, device=self.device)
            for reg_func in self.regularization_functions:
                reg_loss += reg_func(x)
            
            # Compute diversity loss
            diversity_loss = torch.tensor(0.0, device=self.device)
            if diversity_weight > 0 and i > 0:
                # Encourage diversity from previous iterations
                diversity_loss = diversity_weight * torch.norm(x - x_prev)
            
            # Total loss
            total_loss = activation_loss + reg_loss + diversity_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_([x], gradient_clipping)
            
            optimizer.step()
            
            # Clamp values to valid range
            x.data = torch.clamp(x.data, 0, 1)
            
            # Store previous x for diversity
            if diversity_weight > 0:
                x_prev = x.clone().detach()
            
            # Print progress
            if verbose and i % 20 == 0:
                logger.info(
                    f"Iteration {i}/{iterations}, "
                    f"Activation: {-activation_loss.item():.4f}, "
                    f"Reg: {reg_loss.item():.4f}, "
                    f"Total: {total_loss.item():.4f}"
                )
        
        return x.detach()
    
    def generate_multiple_patterns(
        self,
        input_shape: Tuple[int, ...],
        num_patterns: int = 5,
        iterations: int = 200,
        learning_rate: float = 0.01,
        optimization_method: str = "adam",
        diversity_weight: float = 0.1,
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate multiple diverse activation patterns.
        
        Args:
            input_shape: Shape of input tensor.
            num_patterns: Number of patterns to generate.
            iterations: Number of optimization iterations.
            learning_rate: Learning rate.
            optimization_method: Optimization method.
            diversity_weight: Weight for diversity regularization.
            **kwargs: Additional arguments for maximize_activation.
            
        Returns:
            List of optimized input tensors.
        """
        patterns = []
        
        for i in range(num_patterns):
            logger.info(f"Generating pattern {i+1}/{num_patterns}")
            
            # Generate pattern
            pattern = self.maximize_activation(
                input_shape=input_shape,
                iterations=iterations,
                learning_rate=learning_rate,
                optimization_method=optimization_method,
                diversity_weight=diversity_weight,
                verbose=False,
                **kwargs
            )
            
            patterns.append(pattern)
        
        return patterns


class FeatureVisualizer:
    """Feature visualization using activation maximization."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu')
    ):
        """Initialize feature visualizer.
        
        Args:
            model: Neural network model.
            device: Device to run on.
        """
        self.model = model.to(device)
        self.device = device
        
        logger.info("Initialized FeatureVisualizer")
    
    def visualize_layer_features(
        self,
        layer_name: str,
        input_shape: Tuple[int, ...],
        num_features: int = 16,
        iterations: int = 200,
        learning_rate: float = 0.01,
        tv_weight: float = 0.1,
        l2_weight: float = 0.01
    ) -> torch.Tensor:
        """Visualize features learned by a specific layer.
        
        Args:
            layer_name: Name of layer to visualize.
            input_shape: Shape of input tensor.
            num_features: Number of features to visualize.
            iterations: Number of optimization iterations.
            learning_rate: Learning rate.
            tv_weight: Total variation regularization weight.
            l2_weight: L2 regularization weight.
            
        Returns:
            Tensor containing visualized features.
        """
        # Set up regularization
        regularization_functions = [
            TotalVariationRegularization(tv_weight),
            L2Regularization(l2_weight)
        ]
        
        # Get layer output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape, device=self.device)
            layer_output = self.model.get_layer_output(dummy_input, layer_name)
            num_channels = layer_output.size(1)
        
        # Visualize each feature
        feature_visualizations = []
        
        for feature_idx in range(min(num_features, num_channels)):
            logger.info(f"Visualizing feature {feature_idx+1}/{min(num_features, num_channels)}")
            
            # Create activation maximizer for this feature
            maximizer = ActivationMaximizer(
                model=self.model,
                target_layer=layer_name,
                target_class=None,  # We'll maximize specific feature
                regularization_functions=regularization_functions,
                device=self.device
            )
            
            # Custom maximization for specific feature
            x = torch.randn(1, *input_shape, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=learning_rate)
            
            for i in range(iterations):
                optimizer.zero_grad()
                
                # Get layer output
                layer_output = self.model.get_layer_output(x, layer_name)
                
                # Maximize specific feature
                activation_loss = -layer_output[:, feature_idx].mean()
                
                # Add regularization
                reg_loss = torch.tensor(0.0, device=self.device)
                for reg_func in regularization_functions:
                    reg_loss += reg_func(x)
                
                total_loss = activation_loss + reg_loss
                total_loss.backward()
                optimizer.step()
                
                # Clamp values
                x.data = torch.clamp(x.data, 0, 1)
            
            feature_visualizations.append(x.detach())
        
        # Stack visualizations
        return torch.cat(feature_visualizations, dim=0)
    
    def compare_layers(
        self,
        layer_names: List[str],
        input_shape: Tuple[int, ...],
        target_class: int = 1,
        iterations: int = 200,
        learning_rate: float = 0.01,
        tv_weight: float = 0.1,
        l2_weight: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """Compare activation patterns across different layers.
        
        Args:
            layer_names: List of layer names to compare.
            input_shape: Shape of input tensor.
            target_class: Target class for maximization.
            iterations: Number of optimization iterations.
            learning_rate: Learning rate.
            tv_weight: Total variation regularization weight.
            l2_weight: L2 regularization weight.
            
        Returns:
            Dictionary mapping layer names to optimized inputs.
        """
        results = {}
        
        # Set up regularization
        regularization_functions = [
            TotalVariationRegularization(tv_weight),
            L2Regularization(l2_weight)
        ]
        
        for layer_name in layer_names:
            logger.info(f"Maximizing activation for layer: {layer_name}")
            
            maximizer = ActivationMaximizer(
                model=self.model,
                target_layer=layer_name,
                target_class=target_class,
                regularization_functions=regularization_functions,
                device=self.device
            )
            
            pattern = maximizer.maximize_activation(
                input_shape=input_shape,
                iterations=iterations,
                learning_rate=learning_rate,
                verbose=False
            )
            
            results[layer_name] = pattern
        
        return results
