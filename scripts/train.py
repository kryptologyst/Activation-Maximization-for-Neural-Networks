#!/usr/bin/env python3
"""Training script for activation maximization XAI project."""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils import set_seed, get_device, EarlyStopping, save_checkpoint
from src.data import create_data_loaders, get_dataset_info
from src.models import create_model
from src.methods import ActivationMaximizer, FeatureVisualizer
from src.eval import ActivationMaximizationEvaluator
from src.viz import ActivationMaximizationVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device
) -> nn.Module:
    """Train the model.
    
    Args:
        model: Neural network model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration dictionary.
        device: Device to train on.
        
    Returns:
        Trained model.
    """
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        # Calculate average losses and accuracies
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log epoch results
        logger.info(
            f'Epoch {epoch+1}/{config["training"]["epochs"]}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
        )
        
        # Save best model
        if val_loss < best_val_loss and config['training']['save_best_model']:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                Path('checkpoints') / 'best_model.pth',
                {'train_acc': train_acc, 'val_acc': val_acc}
            )
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate the trained model.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to evaluate on.
        
    Returns:
        Evaluation metrics.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }


def run_activation_maximization(
    model: nn.Module,
    config: dict,
    device: torch.device,
    dataset_info: dict
) -> dict:
    """Run activation maximization experiments.
    
    Args:
        model: Trained model.
        config: Configuration dictionary.
        device: Device to run on.
        dataset_info: Dataset information.
        
    Returns:
        Activation maximization results.
    """
    logger.info("Starting activation maximization experiments")
    
    # Initialize visualizer
    visualizer = ActivationMaximizationVisualizer()
    
    # Get input shape
    input_shape = (dataset_info['channels'], *dataset_info['image_size'])
    
    # Set up regularization
    from src.methods import TotalVariationRegularization, L2Regularization
    regularization_functions = [
        TotalVariationRegularization(config['activation_maximization']['regularization']['tv_weight']),
        L2Regularization(config['activation_maximization']['regularization']['l2_weight'])
    ]
    
    # Run activation maximization for different layers
    layer_results = {}
    target_layer = config['activation_maximization']['target_layer']
    target_class = config['activation_maximization']['target_class']
    
    logger.info(f"Maximizing activation for layer '{target_layer}', class {target_class}")
    
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
        iterations=config['activation_maximization']['iterations'],
        learning_rate=config['activation_maximization']['learning_rate'],
        optimization_method=config['activation_maximization']['optimization_method'],
        gradient_clipping=config['activation_maximization']['gradient_clipping']
    )
    
    layer_results[target_layer] = pattern
    
    # Visualize results
    visualizer.visualize_single_pattern(
        pattern,
        title=f"Activation Maximization - Layer: {target_layer}, Class: {target_class}",
        save_path=f"assets/visualizations/activation_pattern_{target_layer}_class_{target_class}.png"
    )
    
    # Generate multiple patterns for diversity
    logger.info("Generating multiple diverse patterns")
    patterns = maximizer.generate_multiple_patterns(
        input_shape=input_shape,
        num_patterns=5,
        iterations=config['activation_maximization']['iterations'],
        learning_rate=config['activation_maximization']['learning_rate'],
        diversity_weight=0.1
    )
    
    visualizer.visualize_multiple_patterns(
        patterns,
        titles=[f"Pattern {i+1}" for i in range(len(patterns))],
        save_path="assets/visualizations/multiple_patterns.png"
    )
    
    # Feature visualization
    logger.info("Generating feature visualizations")
    feature_visualizer = FeatureVisualizer(model, device)
    
    feature_visualizations = feature_visualizer.visualize_layer_features(
        layer_name=target_layer,
        input_shape=input_shape,
        num_features=16,
        iterations=config['activation_maximization']['iterations'],
        learning_rate=config['activation_maximization']['learning_rate'],
        tv_weight=config['activation_maximization']['regularization']['tv_weight'],
        l2_weight=config['activation_maximization']['regularization']['l2_weight']
    )
    
    visualizer.visualize_feature_grid(
        feature_visualizations,
        save_path="assets/visualizations/feature_grid.png"
    )
    
    return {
        'layer_results': layer_results,
        'multiple_patterns': patterns,
        'feature_visualizations': feature_visualizations
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train model and run activation maximization')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only run activation maximization')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['device']['preferred'])
    
    # Get dataset info
    dataset_info = get_dataset_info(config['data']['dataset'])
    
    # Create data loaders
    logger.info(f"Loading {config['data']['dataset']} dataset")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        image_size=dataset_info['image_size'],
        num_classes=dataset_info['num_classes']
    )
    
    # Create model
    logger.info(f"Creating {config['model']['architecture']} model")
    model = create_model(
        architecture=config['model']['architecture'],
        input_channels=dataset_info['channels'],
        hidden_dims=config['model']['hidden_dims'],
        num_classes=dataset_info['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    model = model.to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train model
    if not args.skip_training:
        logger.info("Starting model training")
        model = train_model(model, train_loader, val_loader, config, device)
        
        # Evaluate model
        logger.info("Evaluating model")
        eval_results = evaluate_model(model, test_loader, device)
        logger.info(f"Final test accuracy: {eval_results['test_accuracy']:.2f}%")
    
    # Run activation maximization
    logger.info("Running activation maximization experiments")
    am_results = run_activation_maximization(model, config, device, dataset_info)
    
    logger.info("Training and activation maximization completed successfully!")


if __name__ == '__main__':
    main()
