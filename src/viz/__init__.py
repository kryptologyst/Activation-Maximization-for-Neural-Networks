"""Visualization utilities for activation maximization results."""

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ActivationMaximizationVisualizer:
    """Visualizer for activation maximization results."""
    
    def __init__(self, save_dir: str = "assets/visualizations"):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"Initialized ActivationMaximizationVisualizer, saving to {self.save_dir}")
    
    def visualize_single_pattern(
        self,
        pattern: torch.Tensor,
        title: str = "Activation Maximization Pattern",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """Visualize a single activation maximization pattern.
        
        Args:
            pattern: Generated pattern tensor.
            title: Plot title.
            save_path: Path to save figure.
            figsize: Figure size.
        """
        # Convert to numpy and handle different input shapes
        if pattern.dim() == 4:  # Batch dimension
            pattern = pattern[0]
        
        if pattern.dim() == 3:  # Channel dimension
            if pattern.size(0) == 1:  # Grayscale
                img = pattern[0].cpu().numpy()
                cmap = 'gray'
            else:  # RGB
                img = pattern.permute(1, 2, 0).cpu().numpy()
                cmap = None
        else:
            img = pattern.cpu().numpy()
            cmap = 'gray'
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot image
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_multiple_patterns(
        self,
        patterns: List[torch.Tensor],
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Visualize multiple activation maximization patterns.
        
        Args:
            patterns: List of generated patterns.
            titles: List of titles for each pattern.
            save_path: Path to save figure.
            figsize: Figure size.
        """
        num_patterns = len(patterns)
        cols = min(5, num_patterns)
        rows = (num_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, pattern in enumerate(patterns):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Convert to numpy
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
            
            # Plot
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(titles[i] if titles else f"Pattern {i+1}", fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_patterns, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-pattern visualization to {save_path}")
        
        plt.show()
    
    def visualize_layer_comparison(
        self,
        layer_results: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8)
    ) -> None:
        """Visualize activation patterns across different layers.
        
        Args:
            layer_results: Dictionary mapping layer names to patterns.
            save_path: Path to save figure.
            figsize: Figure size.
        """
        num_layers = len(layer_results)
        fig, axes = plt.subplots(1, num_layers, figsize=figsize)
        
        if num_layers == 1:
            axes = [axes]
        
        for i, (layer_name, pattern) in enumerate(layer_results.items()):
            ax = axes[i]
            
            # Convert to numpy
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
            
            # Plot
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"Layer: {layer_name}", fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved layer comparison to {save_path}")
        
        plt.show()
    
    def visualize_feature_grid(
        self,
        feature_visualizations: torch.Tensor,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 12)
    ) -> None:
        """Visualize a grid of feature visualizations.
        
        Args:
            feature_visualizations: Tensor of feature visualizations.
            save_path: Path to save figure.
            figsize: Figure size.
        """
        num_features = feature_visualizations.size(0)
        cols = int(np.ceil(np.sqrt(num_features)))
        rows = int(np.ceil(num_features / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_features):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Convert to numpy
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
            
            # Plot
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"Feature {i+1}", fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_features, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature grid to {save_path}")
        
        plt.show()
    
    def visualize_evaluation_metrics(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Visualize evaluation metrics as heatmap.
        
        Args:
            evaluation_results: Dictionary mapping method names to metric scores.
            save_path: Path to save figure.
            figsize: Figure size.
        """
        # Prepare data for heatmap
        methods = list(evaluation_results.keys())
        metrics = list(evaluation_results[methods[0]].keys())
        
        data = np.zeros((len(methods), len(metrics)))
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                data[i, j] = evaluation_results[method][metric]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="white" if data[i, j] < 0.5 else "black")
        
        ax.set_title('Activation Maximization Evaluation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved evaluation metrics to {save_path}")
        
        plt.show()
    
    def create_interactive_plot(
        self,
        patterns: List[torch.Tensor],
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive plotly visualization.
        
        Args:
            patterns: List of generated patterns.
            titles: List of titles for each pattern.
            save_path: Path to save HTML file.
            
        Returns:
            Plotly figure.
        """
        num_patterns = len(patterns)
        cols = min(3, num_patterns)
        rows = (num_patterns + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles or [f"Pattern {i+1}" for i in range(num_patterns)],
            specs=[[{"type": "image"} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, pattern in enumerate(patterns):
            row = i // cols + 1
            col = i % cols + 1
            
            # Convert to numpy
            if pattern.dim() == 4:
                pattern = pattern[0]
            
            if pattern.dim() == 3:
                if pattern.size(0) == 1:
                    img = pattern[0].cpu().numpy()
                else:
                    img = pattern.permute(1, 2, 0).cpu().numpy()
            else:
                img = pattern.cpu().numpy()
            
            fig.add_trace(
                go.Image(z=img),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Interactive Activation Maximization Patterns",
            height=400 * rows,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive plot to {save_path}")
        
        return fig
    
    def save_patterns_as_images(
        self,
        patterns: Union[torch.Tensor, List[torch.Tensor]],
        prefix: str = "pattern",
        format: str = "png"
    ) -> List[str]:
        """Save patterns as individual image files.
        
        Args:
            patterns: Pattern(s) to save.
            prefix: Filename prefix.
            format: Image format ('png', 'jpg', etc.).
            
        Returns:
            List of saved file paths.
        """
        if isinstance(patterns, torch.Tensor):
            patterns = [patterns]
        
        saved_paths = []
        
        for i, pattern in enumerate(patterns):
            # Convert to numpy
            if pattern.dim() == 4:
                pattern = pattern[0]
            
            if pattern.dim() == 3:
                if pattern.size(0) == 1:
                    img = pattern[0].cpu().numpy()
                else:
                    img = pattern.permute(1, 2, 0).cpu().numpy()
            else:
                img = pattern.cpu().numpy()
            
            # Save image
            filename = f"{prefix}_{i+1}.{format}"
            filepath = self.save_dir / filename
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            plt.axis('off')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(str(filepath))
        
        logger.info(f"Saved {len(saved_paths)} patterns as images")
        return saved_paths
