"""Streamlit demo for Activation Maximization XAI project."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml
import logging

from src.utils import set_seed, get_device
from src.data import create_data_loaders, get_dataset_info
from src.models import create_model
from src.methods import ActivationMaximizer, FeatureVisualizer, TotalVariationRegularization, L2Regularization
from src.eval import ActivationMaximizationEvaluator
from src.viz import ActivationMaximizationVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Activation Maximization XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This is a research and educational tool only.</strong> Activation maximization outputs may be unstable, misleading, or not representative of actual model behavior. 
    Do not use these visualizations for regulated decisions without human review and validation.</p>
    <p>XAI methods are experimental and should be interpreted with caution. Always verify results with domain experts.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧠 Activation Maximization for Neural Networks</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Load configuration
@st.cache_data
def load_config():
    """Load configuration file."""
    try:
        with open('configs/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure configs/config.yaml exists.")
        st.stop()

config = load_config()

# Dataset selection
dataset_options = {
    "MNIST": "mnist",
    "CIFAR-10": "cifar10",
    "Synthetic": "synthetic"
}
selected_dataset = st.sidebar.selectbox("Dataset", list(dataset_options.keys()))
dataset_name = dataset_options[selected_dataset]

# Get dataset info
dataset_info = get_dataset_info(dataset_name)

# Model configuration
st.sidebar.subheader("Model Configuration")
architecture = st.sidebar.selectbox("Architecture", ["simple_cnn", "resnet18"])
target_layer = st.sidebar.selectbox("Target Layer", ["conv_1", "conv_2", "conv_3", "fc1", "fc2"])
target_class = st.sidebar.selectbox("Target Class", list(range(dataset_info['num_classes'])))

# Activation maximization parameters
st.sidebar.subheader("Activation Maximization Parameters")
iterations = st.sidebar.slider("Iterations", 50, 500, 200)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
tv_weight = st.sidebar.slider("TV Regularization Weight", 0.0, 1.0, 0.1, 0.01)
l2_weight = st.sidebar.slider("L2 Regularization Weight", 0.0, 0.1, 0.01, 0.001)
optimization_method = st.sidebar.selectbox("Optimization Method", ["adam", "sgd", "rmsprop"])

# Device selection
device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if torch.backends.mps.is_available():
    device_options.append("mps")

selected_device = st.sidebar.selectbox("Device", device_options)
if selected_device == "auto":
    device = get_device()
else:
    device = torch.device(selected_device)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaders' not in st.session_state:
    st.session_state.data_loaders = None

# Load model button
if st.sidebar.button("Load Model", type="primary"):
    with st.spinner("Loading model and data..."):
        try:
            # Set seed for reproducibility
            set_seed(42)
            
            # Create model
            model = create_model(
                architecture=architecture,
                input_channels=dataset_info['channels'],
                hidden_dims=[32, 64, 128],
                num_classes=dataset_info['num_classes'],
                dropout_rate=0.2
            )
            model = model.to(device)
            model.eval()
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset_name=dataset_name,
                data_dir="data",
                batch_size=64,
                num_workers=0,  # Streamlit doesn't support multiprocessing
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                image_size=dataset_info['image_size'],
                num_classes=dataset_info['num_classes']
            )
            
            st.session_state.model = model
            st.session_state.data_loaders = (train_loader, val_loader, test_loader)
            
            st.success("Model and data loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Main content
if st.session_state.model is not None:
    model = st.session_state.model
    train_loader, val_loader, test_loader = st.session_state.data_loaders
    
    # Model information
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Architecture", architecture)
    with col2:
        st.metric("Parameters", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    with col3:
        st.metric("Device", str(device))
    
    # Activation maximization section
    st.subheader("Activation Maximization")
    
    if st.button("Generate Activation Pattern", type="primary"):
        with st.spinner("Generating activation maximization pattern..."):
            try:
                # Set up regularization
                regularization_functions = [
                    TotalVariationRegularization(tv_weight),
                    L2Regularization(l2_weight)
                ]
                
                # Get input shape
                input_shape = (dataset_info['channels'], *dataset_info['image_size'])
                
                # Create activation maximizer
                maximizer = ActivationMaximizer(
                    model=model,
                    target_layer=target_layer,
                    target_class=target_class,
                    regularization_functions=regularization_functions,
                    device=device
                )
                
                # Generate pattern
                pattern = maximizer.maximize_activation(
                    input_shape=input_shape,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    optimization_method=optimization_method,
                    verbose=False
                )
                
                # Display results
                st.success("Activation pattern generated successfully!")
                
                # Convert pattern for display
                if pattern.dim() == 4:
                    pattern_display = pattern[0]
                else:
                    pattern_display = pattern
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(8, 8))
                
                if pattern_display.dim() == 3:
                    if pattern_display.size(0) == 1:
                        img = pattern_display[0].cpu().numpy()
                        cmap = 'gray'
                    else:
                        img = pattern_display.permute(1, 2, 0).cpu().numpy()
                        cmap = None
                else:
                    img = pattern_display.cpu().numpy()
                    cmap = 'gray'
                
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f"Activation Maximization Pattern\nLayer: {target_layer}, Class: {target_class}")
                ax.axis('off')
                
                st.pyplot(fig)
                
                # Evaluation metrics
                st.subheader("Evaluation Metrics")
                
                evaluator = ActivationMaximizationEvaluator()
                
                # Get some test data for evaluation
                test_data, test_labels = next(iter(test_loader))
                test_data = test_data[:10]  # Use first 10 samples
                test_labels = test_labels[:10]
                
                metrics = evaluator.evaluate(
                    model=model,
                    generated_input=pattern,
                    target_layer=target_layer,
                    target_class=target_class,
                    test_data=test_data,
                    test_labels=test_labels
                )
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Activation Strength", f"{metrics.get('ActivationStrengthMetric', 0):.4f}")
                with col2:
                    st.metric("Visual Quality (Sharpness)", f"{metrics.get('VisualQualityMetric', 0):.4f}")
                with col3:
                    st.metric("Faithfulness", f"{metrics.get('FaithfulnessMetric', 0):.4f}")
                with col4:
                    st.metric("Pattern Diversity", f"{metrics.get('PatternDiversityMetric', 0):.4f}")
                
                # Generate multiple patterns
                st.subheader("Multiple Patterns")
                
                if st.button("Generate Multiple Patterns"):
                    with st.spinner("Generating multiple diverse patterns..."):
                        patterns = maximizer.generate_multiple_patterns(
                            input_shape=input_shape,
                            num_patterns=5,
                            iterations=iterations,
                            learning_rate=learning_rate,
                            optimization_method=optimization_method,
                            diversity_weight=0.1
                        )
                        
                        # Display patterns in a grid
                        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                        
                        for i, pattern in enumerate(patterns):
                            ax = axes[i]
                            
                            if pattern.dim() == 4:
                                pattern = pattern[0]
                            
                            if pattern.dim() == 3:
                                if pattern.size(0) == 1:
                                    img = pattern[0].cpu().numpy()
                                    cmap = 'gray'
                                else:
                                    img = pattern.permute(1, 2, 0).cpu().numpy()
                                    cmap = None
                            else:
                                img = pattern.cpu().numpy()
                                cmap = 'gray'
                            
                            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                            ax.set_title(f"Pattern {i+1}")
                            ax.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Feature visualization
                st.subheader("Feature Visualization")
                
                if st.button("Generate Feature Grid"):
                    with st.spinner("Generating feature visualizations..."):
                        feature_visualizer = FeatureVisualizer(model, device)
                        
                        feature_visualizations = feature_visualizer.visualize_layer_features(
                            layer_name=target_layer,
                            input_shape=input_shape,
                            num_features=16,
                            iterations=iterations,
                            learning_rate=learning_rate,
                            tv_weight=tv_weight,
                            l2_weight=l2_weight
                        )
                        
                        # Display feature grid
                        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                        
                        for i in range(16):
                            row = i // 4
                            col = i % 4
                            ax = axes[row, col]
                            
                            pattern = feature_visualizations[i]
                            if pattern.dim() == 3:
                                if pattern.size(0) == 1:
                                    img = pattern[0].cpu().numpy()
                                    cmap = 'gray'
                                else:
                                    img = pattern.permute(1, 2, 0).cpu().numpy()
                                    cmap = None
                            else:
                                img = pattern.cpu().numpy()
                                cmap = 'gray'
                            
                            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                            ax.set_title(f"Feature {i+1}")
                            ax.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating activation pattern: {str(e)}")
                logger.error(f"Activation maximization error: {str(e)}")
    
    # Layer comparison
    st.subheader("Layer Comparison")
    
    if st.button("Compare Layers"):
        with st.spinner("Comparing activation patterns across layers..."):
            try:
                available_layers = model.get_layer_names()
                
                # Generate patterns for different layers
                layer_results = {}
                
                for layer in available_layers[:3]:  # Compare first 3 layers
                    maximizer = ActivationMaximizer(
                        model=model,
                        target_layer=layer,
                        target_class=target_class,
                        regularization_functions=[
                            TotalVariationRegularization(tv_weight),
                            L2Regularization(l2_weight)
                        ],
                        device=device
                    )
                    
                    pattern = maximizer.maximize_activation(
                        input_shape=input_shape,
                        iterations=iterations,
                        learning_rate=learning_rate,
                        optimization_method=optimization_method,
                        verbose=False
                    )
                    
                    layer_results[layer] = pattern
                
                # Display comparison
                fig, axes = plt.subplots(1, len(layer_results), figsize=(15, 5))
                
                if len(layer_results) == 1:
                    axes = [axes]
                
                for i, (layer_name, pattern) in enumerate(layer_results.items()):
                    ax = axes[i]
                    
                    if pattern.dim() == 4:
                        pattern = pattern[0]
                    
                    if pattern.dim() == 3:
                        if pattern.size(0) == 1:
                            img = pattern[0].cpu().numpy()
                            cmap = 'gray'
                        else:
                            img = pattern.permute(1, 2, 0).cpu().numpy()
                            cmap = None
                    else:
                        img = pattern.cpu().numpy()
                        cmap = 'gray'
                    
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                    ax.set_title(f"Layer: {layer_name}")
                    ax.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error comparing layers: {str(e)}")
                logger.error(f"Layer comparison error: {str(e)}")

else:
    st.info("Please load a model first using the sidebar controls.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Activation Maximization XAI Project | Research and Educational Use Only</p>
    <p>⚠️ Results may be unstable or misleading - use with caution</p>
</div>
""", unsafe_allow_html=True)
