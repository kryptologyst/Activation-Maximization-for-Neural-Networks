"""Evaluation metrics for activation maximization."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ActivationMaximizationMetric(ABC):
    """Abstract base class for activation maximization metrics."""
    
    @abstractmethod
    def compute(self, model: nn.Module, generated_input: torch.Tensor, target_layer: str, target_class: Optional[int] = None) -> float:
        """Compute metric value.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            
        Returns:
            Metric value.
        """
        pass


class ActivationStrengthMetric(ActivationMaximizationMetric):
    """Measure the strength of activation achieved."""
    
    def compute(self, model: nn.Module, generated_input: torch.Tensor, target_layer: str, target_class: Optional[int] = None) -> float:
        """Compute activation strength.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            
        Returns:
            Activation strength value.
        """
        with torch.no_grad():
            layer_output = model.get_layer_output(generated_input, target_layer)
            
            if target_class is not None:
                activation = layer_output[:, target_class].mean().item()
            else:
                activation = layer_output.mean().item()
        
        return activation


class PatternDiversityMetric(ActivationMaximizationMetric):
    """Measure diversity among generated patterns."""
    
    def __init__(self, num_patterns: int = 5):
        """Initialize diversity metric.
        
        Args:
            num_patterns: Number of patterns to generate for diversity measurement.
        """
        self.num_patterns = num_patterns
    
    def compute(self, model: nn.Module, generated_input: torch.Tensor, target_layer: str, target_class: Optional[int] = None) -> float:
        """Compute pattern diversity.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            
        Returns:
            Diversity score (higher is more diverse).
        """
        # This is a placeholder - in practice, you'd generate multiple patterns
        # and measure their pairwise differences
        return 0.0


class VisualQualityMetric(ActivationMaximizationMetric):
    """Measure visual quality of generated patterns."""
    
    def __init__(self, method: str = "sharpness"):
        """Initialize visual quality metric.
        
        Args:
            method: Quality measurement method ('sharpness', 'contrast', 'naturalness').
        """
        self.method = method
    
    def compute(self, model: nn.Module, generated_input: torch.Tensor, target_layer: str, target_class: Optional[int] = None) -> float:
        """Compute visual quality.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            
        Returns:
            Visual quality score.
        """
        if self.method == "sharpness":
            return self._compute_sharpness(generated_input)
        elif self.method == "contrast":
            return self._compute_contrast(generated_input)
        elif self.method == "naturalness":
            return self._compute_naturalness(generated_input)
        else:
            raise ValueError(f"Unknown quality method: {self.method}")
    
    def _compute_sharpness(self, image: torch.Tensor) -> float:
        """Compute image sharpness using Laplacian variance."""
        # Convert to grayscale if needed
        if image.size(1) > 1:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # Compute Laplacian
        laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=image.device)
        laplacian = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)
        
        # Compute variance
        sharpness = torch.var(laplacian).item()
        return sharpness
    
    def _compute_contrast(self, image: torch.Tensor) -> float:
        """Compute image contrast."""
        # Convert to grayscale if needed
        if image.size(1) > 1:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # Compute standard deviation as contrast measure
        contrast = torch.std(gray).item()
        return contrast
    
    def _compute_naturalness(self, image: torch.Tensor) -> float:
        """Compute naturalness score (placeholder implementation)."""
        # This would typically involve comparing to natural image statistics
        # For now, return a simple measure based on pixel value distribution
        return torch.mean(image).item()


class FaithfulnessMetric(ActivationMaximizationMetric):
    """Measure faithfulness of generated patterns to model behavior."""
    
    def __init__(self, test_data: torch.Tensor, test_labels: torch.Tensor):
        """Initialize faithfulness metric.
        
        Args:
            test_data: Test dataset.
            test_labels: Test labels.
        """
        self.test_data = test_data
        self.test_labels = test_labels
    
    def compute(self, model: nn.Module, generated_input: torch.Tensor, target_layer: str, target_class: Optional[int] = None) -> float:
        """Compute faithfulness score.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            
        Returns:
            Faithfulness score.
        """
        with torch.no_grad():
            # Get activations for generated input
            gen_activation = model.get_layer_output(generated_input, target_layer)
            
            # Get activations for test data
            test_activations = []
            for i in range(min(100, len(self.test_data))):  # Sample subset for efficiency
                test_input = self.test_data[i:i+1]
                activation = model.get_layer_output(test_input, target_layer)
                test_activations.append(activation)
            
            test_activations = torch.cat(test_activations, dim=0)
            
            # Compute similarity (higher is more faithful)
            if target_class is not None:
                gen_val = gen_activation[:, target_class].mean()
                test_vals = test_activations[:, target_class].mean()
            else:
                gen_val = gen_activation.mean()
                test_vals = test_activations.mean()
            
            # Normalize similarity
            similarity = 1.0 - abs(gen_val - test_vals) / (abs(gen_val) + abs(test_vals) + 1e-8)
            return similarity.item()


class StabilityMetric(ActivationMaximizationMetric):
    """Measure stability of activation maximization across different initializations."""
    
    def __init__(self, num_runs: int = 5):
        """Initialize stability metric.
        
        Args:
            num_runs: Number of runs to compute stability.
        """
        self.num_runs = num_runs
    
    def compute(self, model: nn.Module, generated_input: torch.Tensor, target_layer: str, target_class: Optional[int] = None) -> float:
        """Compute stability score.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            
        Returns:
            Stability score (higher is more stable).
        """
        # This is a placeholder - in practice, you'd run activation maximization
        # multiple times with different initializations and measure consistency
        return 0.0


class ActivationMaximizationEvaluator:
    """Comprehensive evaluator for activation maximization methods."""
    
    def __init__(self, metrics: Optional[List[ActivationMaximizationMetric]] = None):
        """Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute.
        """
        self.metrics = metrics or [
            ActivationStrengthMetric(),
            VisualQualityMetric("sharpness"),
            VisualQualityMetric("contrast")
        ]
        
        logger.info(f"Initialized ActivationMaximizationEvaluator with {len(self.metrics)} metrics")
    
    def evaluate(
        self,
        model: nn.Module,
        generated_input: torch.Tensor,
        target_layer: str,
        target_class: Optional[int] = None,
        test_data: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate activation maximization results.
        
        Args:
            model: Neural network model.
            generated_input: Generated input from activation maximization.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            test_data: Test data for faithfulness evaluation.
            test_labels: Test labels for faithfulness evaluation.
            
        Returns:
            Dictionary of metric scores.
        """
        results = {}
        
        for metric in self.metrics:
            try:
                # Add test data to faithfulness metric if available
                if isinstance(metric, FaithfulnessMetric) and test_data is not None and test_labels is not None:
                    metric.test_data = test_data
                    metric.test_labels = test_labels
                
                score = metric.compute(model, generated_input, target_layer, target_class)
                results[metric.__class__.__name__] = score
                
            except Exception as e:
                logger.warning(f"Failed to compute metric {metric.__class__.__name__}: {e}")
                results[metric.__class__.__name__] = 0.0
        
        logger.info(f"Computed {len(results)} evaluation metrics")
        return results
    
    def compare_methods(
        self,
        model: nn.Module,
        generated_inputs: Dict[str, torch.Tensor],
        target_layer: str,
        target_class: Optional[int] = None,
        test_data: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare different activation maximization methods.
        
        Args:
            model: Neural network model.
            generated_inputs: Dictionary mapping method names to generated inputs.
            target_layer: Target layer name.
            target_class: Target class (if applicable).
            test_data: Test data for faithfulness evaluation.
            test_labels: Test labels for faithfulness evaluation.
            
        Returns:
            Dictionary mapping method names to metric scores.
        """
        results = {}
        
        for method_name, generated_input in generated_inputs.items():
            logger.info(f"Evaluating method: {method_name}")
            
            method_results = self.evaluate(
                model=model,
                generated_input=generated_input,
                target_layer=target_layer,
                target_class=target_class,
                test_data=test_data,
                test_labels=test_labels
            )
            
            results[method_name] = method_results
        
        return results
    
    def create_leaderboard(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metric_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Create a leaderboard ranking methods by overall score.
        
        Args:
            comparison_results: Results from compare_methods.
            metric_weights: Weights for different metrics.
            
        Returns:
            Dictionary mapping method names to overall scores.
        """
        if metric_weights is None:
            metric_weights = {metric.__class__.__name__: 1.0 for metric in self.metrics}
        
        overall_scores = {}
        
        for method_name, method_results in comparison_results.items():
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric_name, score in method_results.items():
                weight = metric_weights.get(metric_name, 1.0)
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_scores[method_name] = weighted_score / total_weight
            else:
                overall_scores[method_name] = 0.0
        
        # Sort by score (descending)
        sorted_scores = dict(sorted(overall_scores.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("Created activation maximization leaderboard")
        return sorted_scores
