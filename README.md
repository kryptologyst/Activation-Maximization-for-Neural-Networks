# Activation Maximization for Neural Networks

Production-ready implementation of activation maximization for neural network interpretability, focusing on research and educational applications.

## ⚠️ Important Disclaimer

**This is a research and educational tool only.** Activation maximization outputs may be unstable, misleading, or not representative of actual model behavior. Do not use these visualizations for regulated decisions without human review and validation.

XAI methods are experimental and should be interpreted with caution. Always verify results with domain experts.

## Overview

Activation maximization is a technique used to visualize and understand what a neural network is learning by maximizing the activation of specific neurons or layers. This project provides a comprehensive implementation with:

- Modern PyTorch-based implementation
- Multiple regularization techniques (TV, L2, blur)
- Comprehensive evaluation metrics
- Interactive visualization tools
- Production-ready code structure

## Features

### Core Functionality
- **Activation Maximization**: Generate input patterns that maximize specific neuron/layer activations
- **Feature Visualization**: Visualize learned features across different layers
- **Multiple Regularization**: Total variation, L2, and blur regularization for natural-looking patterns
- **Layer Comparison**: Compare activation patterns across different network layers

### Evaluation Metrics
- **Activation Strength**: Measure the strength of achieved activations
- **Visual Quality**: Assess sharpness, contrast, and naturalness of generated patterns
- **Faithfulness**: Evaluate how well patterns represent actual model behavior
- **Pattern Diversity**: Measure diversity among multiple generated patterns
- **Stability**: Assess consistency across different initializations

### Visualization Tools
- **Interactive Demo**: Streamlit-based web interface
- **Multiple Pattern Display**: Grid visualization of diverse patterns
- **Feature Grid**: Comprehensive feature visualization
- **Layer Comparison**: Side-by-side comparison of different layers
- **Export Capabilities**: Save visualizations in multiple formats

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Activation-Maximization-for-Neural-Networks.git
cd Activation-Maximization-for-Neural-Networks
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Train a Model

```bash
python scripts/train.py --config configs/config.yaml
```

### 2. Run Activation Maximization

```bash
python scripts/train.py --config configs/config.yaml --skip_training
```

### 3. Launch Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

## Configuration

The project uses YAML configuration files. Key parameters:

```yaml
# Data configuration
data:
  dataset: "mnist"  # Options: mnist, cifar10, synthetic
  batch_size: 64
  image_size: [28, 28]

# Model configuration
model:
  architecture: "simple_cnn"  # Options: simple_cnn, resnet18
  hidden_dims: [32, 64, 128]

# Activation maximization
activation_maximization:
  target_layer: "conv2d_1"
  target_class: 1
  iterations: 200
  learning_rate: 0.01
  regularization:
    tv_weight: 0.1
    l2_weight: 0.01
```

## Usage Examples

### Basic Activation Maximization

```python
from src.methods import ActivationMaximizer, TotalVariationRegularization, L2Regularization
from src.models import create_model

# Create model
model = create_model("simple_cnn")

# Set up regularization
regularization_functions = [
    TotalVariationRegularization(0.1),
    L2Regularization(0.01)
]

# Create maximizer
maximizer = ActivationMaximizer(
    model=model,
    target_layer="conv_1",
    target_class=1,
    regularization_functions=regularization_functions
)

# Generate pattern
pattern = maximizer.maximize_activation(
    input_shape=(1, 28, 28),
    iterations=200,
    learning_rate=0.01
)
```

### Feature Visualization

```python
from src.methods import FeatureVisualizer

# Create visualizer
visualizer = FeatureVisualizer(model)

# Generate feature visualizations
features = visualizer.visualize_layer_features(
    layer_name="conv_1",
    input_shape=(1, 28, 28),
    num_features=16
)
```

### Evaluation

```python
from src.eval import ActivationMaximizationEvaluator

# Create evaluator
evaluator = ActivationMaximizationEvaluator()

# Evaluate pattern
metrics = evaluator.evaluate(
    model=model,
    generated_input=pattern,
    target_layer="conv_1",
    target_class=1
)
```

## Project Structure

```
activation-maximization-xai/
├── src/
│   ├── methods/          # Activation maximization implementations
│   ├── models/           # Neural network architectures
│   ├── data/             # Data loading and preprocessing
│   ├── eval/             # Evaluation metrics
│   ├── viz/              # Visualization utilities
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── demo/                 # Interactive demos
├── tests/                # Unit tests
├── assets/               # Generated visualizations
├── data/                 # Dataset storage
└── notebooks/            # Jupyter notebooks
```

## Datasets

### Supported Datasets
- **MNIST**: Handwritten digit recognition (28x28 grayscale)
- **CIFAR-10**: Natural image classification (32x32 RGB)
- **Synthetic**: Generated data for testing

### Dataset Information
- **MNIST**: 10 classes, 1 channel, 28x28 images
- **CIFAR-10**: 10 classes, 3 channels, 32x32 images
- **Synthetic**: Configurable classes and image sizes

## Models

### Supported Architectures
- **SimpleCNN**: Basic convolutional network for MNIST
- **ResNet-18**: Residual network for CIFAR-10

### Model Features
- Configurable hidden dimensions
- Dropout regularization
- Batch normalization
- Multiple activation functions

## Evaluation

### Metrics Implemented
1. **Activation Strength**: Raw activation value achieved
2. **Visual Quality**: Sharpness, contrast, naturalness measures
3. **Faithfulness**: Consistency with test data behavior
4. **Pattern Diversity**: Variation among multiple patterns
5. **Stability**: Consistency across runs

### Evaluation Process
1. Generate activation patterns
2. Compute multiple metrics
3. Compare across different methods
4. Create leaderboard rankings

## Limitations and Considerations

### Known Limitations
- **Instability**: Activation maximization can produce unstable results
- **Local Optima**: Optimization may get stuck in local minima
- **Interpretability**: Generated patterns may not reflect true model behavior
- **Computational Cost**: Requires significant computational resources

### Best Practices
- Use multiple regularization techniques
- Generate multiple patterns for diversity
- Validate results with domain experts
- Consider computational constraints
- Interpret results with caution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{activation_maximization_xai,
  title={Activation Maximization for Neural Networks},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Activation-Maximization-for-Neural-Networks}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Captum for interpretability tools
- Streamlit for the interactive demo framework
- The XAI research community for inspiration and methods

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the research team
- Check the documentation

---

**Remember**: This tool is for research and educational purposes only. Always validate results with domain experts and use with appropriate caution.
# Activation-Maximization-for-Neural-Networks
