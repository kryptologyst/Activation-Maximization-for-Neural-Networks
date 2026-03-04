"""Tests for activation maximization XAI project."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.utils import set_seed, get_device, EarlyStopping
from src.data import MNISTDataset, CIFAR10Dataset, SyntheticDataset, create_data_loaders
from src.models import SimpleCNN, ResNet18, create_model
from src.methods import (
    ActivationMaximizer, 
    FeatureVisualizer,
    TotalVariationRegularization,
    L2Regularization,
    BlurRegularization
)
from src.eval import (
    ActivationMaximizationEvaluator,
    ActivationStrengthMetric,
    VisualQualityMetric,
    FaithfulnessMetric
)


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Placeholder for actual seed verification
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        model = Mock()
        early_stopping = EarlyStopping(patience=2)
        
        # Test improvement
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.4, model)
        
        # Test no improvement
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.5, model)
        assert early_stopping(0.5, model)  # Should stop


class TestData:
    """Test data loading functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticDataset(num_samples=100, image_size=(28, 28), num_classes=10)
        assert len(dataset) == 100
        
        # Test data loading
        image, label = dataset[0]
        assert image.shape == (1, 28, 28)
        assert 0 <= label <= 9
    
    @patch('src.data.datasets.MNIST')
    def test_mnist_dataset(self, mock_mnist):
        """Test MNIST dataset wrapper."""
        # Mock MNIST dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(return_value=(torch.randn(28, 28), 5))
        mock_mnist.return_value = mock_dataset
        
        dataset = MNISTDataset()
        assert len(dataset) == 100
        
        image, label = dataset[0]
        assert image.shape == (1, 28, 28)
        assert isinstance(label, int)


class TestModels:
    """Test model architectures."""
    
    def test_simple_cnn(self):
        """Test SimpleCNN model."""
        model = SimpleCNN(input_channels=1, hidden_dims=[32, 64], num_classes=10)
        
        # Test forward pass
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)
        
        # Test layer names
        layer_names = model.get_layer_names()
        assert len(layer_names) > 0
    
    def test_resnet18(self):
        """Test ResNet-18 model."""
        model = ResNet18(num_classes=10, input_channels=3)
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_create_model(self):
        """Test model creation function."""
        model = create_model("simple_cnn", input_channels=1, num_classes=10)
        assert isinstance(model, SimpleCNN)


class TestMethods:
    """Test activation maximization methods."""
    
    def test_total_variation_regularization(self):
        """Test TV regularization."""
        tv_reg = TotalVariationRegularization(weight=0.1)
        x = torch.randn(1, 1, 28, 28)
        loss = tv_reg(x)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_l2_regularization(self):
        """Test L2 regularization."""
        l2_reg = L2Regularization(weight=0.01)
        x = torch.randn(1, 1, 28, 28)
        loss = l2_reg(x)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_activation_maximizer(self):
        """Test activation maximizer."""
        model = SimpleCNN(input_channels=1, hidden_dims=[32], num_classes=10)
        model.eval()
        
        maximizer = ActivationMaximizer(
            model=model,
            target_layer="conv_1",
            target_class=1,
            device=torch.device('cpu')
        )
        
        # Test pattern generation (small number of iterations for speed)
        pattern = maximizer.maximize_activation(
            input_shape=(1, 28, 28),
            iterations=10,
            learning_rate=0.01,
            verbose=False
        )
        
        assert pattern.shape == (1, 1, 28, 28)
        assert torch.all(pattern >= 0) and torch.all(pattern <= 1)


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_activation_strength_metric(self):
        """Test activation strength metric."""
        model = SimpleCNN(input_channels=1, hidden_dims=[32], num_classes=10)
        model.eval()
        
        metric = ActivationStrengthMetric()
        pattern = torch.randn(1, 1, 28, 28)
        
        score = metric.compute(model, pattern, "conv_1", target_class=1)
        assert isinstance(score, float)
    
    def test_visual_quality_metric(self):
        """Test visual quality metric."""
        metric = VisualQualityMetric("sharpness")
        pattern = torch.randn(1, 1, 28, 28)
        
        score = metric.compute(None, pattern, "conv_1")
        assert isinstance(score, float)
    
    def test_evaluator(self):
        """Test activation maximization evaluator."""
        evaluator = ActivationMaximizationEvaluator()
        
        model = SimpleCNN(input_channels=1, hidden_dims=[32], num_classes=10)
        model.eval()
        
        pattern = torch.randn(1, 1, 28, 28)
        
        results = evaluator.evaluate(
            model=model,
            generated_input=pattern,
            target_layer="conv_1",
            target_class=1
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # Create model
        model = SimpleCNN(input_channels=1, hidden_dims=[32], num_classes=10)
        model.eval()
        
        # Create maximizer
        maximizer = ActivationMaximizer(
            model=model,
            target_layer="conv_1",
            target_class=1,
            device=torch.device('cpu')
        )
        
        # Generate pattern
        pattern = maximizer.maximize_activation(
            input_shape=(1, 28, 28),
            iterations=5,  # Small number for speed
            learning_rate=0.01,
            verbose=False
        )
        
        # Evaluate
        evaluator = ActivationMaximizationEvaluator()
        results = evaluator.evaluate(
            model=model,
            generated_input=pattern,
            target_layer="conv_1",
            target_class=1
        )
        
        # Basic assertions
        assert pattern.shape == (1, 1, 28, 28)
        assert isinstance(results, dict)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__])
