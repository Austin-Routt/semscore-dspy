"""
Visualization module for SemScore characterization.
Provides plotting functions for analyzing and comparing semantic similarity distributions.

This module implements:
1. Distribution plots across categories
2. Effect size heatmaps
3. Bootstrap confidence intervals
4. Category comparison plots
5. Interactive visualizations for notebooks
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from .statistics import StatisticalSummary
from .distributions import CategoryDistributions, DistributionComparison

logger = logging.getLogger(__name__)

class SemScoreVisualizer:
    """Visualization suite for SemScore analysis."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        style: str = 'ggplot',
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        """
        Initialize visualizer with style settings.
        
        Args:
            output_dir: Directory to save plots (optional)
            style: Matplotlib style to use
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        plt.style.use(style)
        self.figsize = figsize
        self.dpi = dpi
        
        # Set color scheme
        self.colors = sns.color_palette("husl", 8)
        self.cmap = sns.diverging_palette(220, 20, as_cmap=True)

    def plot_category_distributions(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Semantic Score Distributions by Category",
        show_stats: bool = True,
        violin: bool = True
    ) -> plt.Figure:
        """
        Create distribution plot comparing categories.
        
        Args:
            data: Dictionary mapping categories to score arrays
            title: Plot title
            show_stats: Whether to show statistical annotations
            violin: Whether to use violin plots (else KDE)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if violin:
            # Create violin plots
            parts = ax.violinplot(
                [data[cat] for cat in data],
                showmeans=True,
                showextrema=True
            )
            
            # Customize violin appearance
            for pc in parts['bodies']:
                pc.set_facecolor(self.colors[0])
                pc.set_alpha(0.7)
            
            # Set category labels
            ax.set_xticks(range(1, len(data) + 1))
            ax.set_xticklabels(list(data.keys()))
            
        else:
            # Create KDE plots
            for i, (category, scores) in enumerate(data.items()):
                sns.kdeplot(
                    data=scores,
                    label=category,
                    color=self.colors[i],
                    fill=True,
                    alpha=0.3,
                    ax=ax
                )
        
        ax.set_title(title)
        ax.set_xlabel("Semantic Similarity Score")
        ax.set_ylabel("Density" if not violin else "Score Distribution")
        ax.grid(True, alpha=0.3)
        
        if show_stats:
            # Add statistical annotations
            stats_text = []
            for category, scores in data.items():
                stats = f"{category}:\n"
                stats += f"μ={np.mean(scores):.3f}\n"
                stats += f"σ={np.std(scores):.3f}\n"
                stats += f"n={len(scores)}"
                stats_text.append(stats)
                
            ax.text(
                0.95, 0.95,
                "\n\n".join(stats_text),
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(
                self.output_dir / f"category_distributions.png",
                dpi=self.dpi,
                bbox_inches='tight'
            )
            
        return fig

    def plot_effect_size_heatmap(
        self,
        effect_matrix: np.ndarray,
        categories: List[str],
        title: str = "Effect Size Matrix",
        cmap: Optional[str] = None,
        annotate: bool = True
    ) -> plt.Figure:
        """
        Create heatmap of effect sizes between categories.
        
        Args:
            effect_matrix: Matrix of effect sizes
            categories: Category labels
            title: Plot title
            cmap: Optional colormap name
            annotate: Whether to show numerical annotations
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create mask for diagonal
        mask = np.zeros_like(effect_matrix, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # Plot heatmap
        sns.heatmap(
            effect_matrix,
            mask=mask,
            cmap=cmap or self.cmap,
            annot=annotate,
            fmt='.3f',
            square=True,
            xticklabels=categories,
            yticklabels=categories,
            ax=ax
        )
        
        ax.set_title(title)
        
        # Rotate tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(
                self.output_dir / f"effect_size_heatmap.png",
                dpi=self.dpi,
                bbox_inches='tight'
            )
            
        return fig

    def plot_bootstrap_distributions(
        self,
        bootstrap_data: Dict[str, np.ndarray],
        ci_level: float = 0.95,
        show_individual: bool = True
    ) -> plt.Figure:
        """
        Create bootstrap distribution plots with confidence intervals.
        
        Args:
            bootstrap_data: Dictionary mapping categories to bootstrap distributions
            ci_level: Confidence interval level
            show_individual: Whether to show individual category plots
            
        Returns:
            matplotlib Figure object
        """
        if show_individual:
            # Create subplot grid
            n_cats = len(bootstrap_data)
            n_cols = min(3, n_cats)
            n_rows = (n_cats - 1) // n_cols + 2  # +1 for overlay plot
            
            fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * n_rows/2))
            gs = plt.GridSpec(n_rows, n_cols)
            
            # Create overlay plot
            ax_overlay = fig.add_subplot(gs[0, :])
            
            for i, (category, dist) in enumerate(bootstrap_data.items()):
                sns.kdeplot(
                    data=dist,
                    label=category,
                    color=self.colors[i],
                    fill=True,
                    alpha=0.3,
                    ax=ax_overlay
                )
                
            ax_overlay.set_title("Bootstrap Distributions Overlay")
            ax_overlay.legend()
            ax_overlay.grid(True, alpha=0.3)
            
            # Create individual plots
            for idx, (category, dist) in enumerate(bootstrap_data.items()):
                row = 1 + idx // n_cols
                col = idx % n_cols
                
                ax = fig.add_subplot(gs[row, col])
                
                # Plot distribution
                sns.kdeplot(
                    data=dist,
                    color=self.colors[idx],
                    fill=True,
                    ax=ax
                )
                
                # Add CI lines
                ci = np.percentile(dist, [(1-ci_level)*50, 50+(ci_level*50)])
                ax.axvline(ci[0], color='r', linestyle='--', alpha=0.5)
                ax.axvline(ci[1], color='r', linestyle='--', alpha=0.5)
                
                ax.set_title(f"{category}\nCI: [{ci[0]:.3f}, {ci[1]:.3f}]")
                ax.grid(True, alpha=0.3)
                
        else:
            # Single overlay plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            for i, (category, dist) in enumerate(bootstrap_data.items()):
                sns.kdeplot(
                    data=dist,
                    label=category,
                    color=self.colors[i],
                    fill=True,
                    alpha=0.3,
                    ax=ax
                )
                
            ax.set_title("Bootstrap Distributions")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(
                self.output_dir / f"bootstrap_distributions.png",
                dpi=self.dpi,
                bbox_inches='tight'
            )
            
        return fig

    def create_analysis_report(
        self,
        results: CategoryDistributions,
        scores: Dict[str, np.ndarray],
        bootstrap_data: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Create comprehensive visualization report.
        
        Args:
            results: Analysis results from CategoryDistributions
            scores: Raw score data by category
            bootstrap_data: Optional bootstrap distribution data
            
        Creates multiple plots and saves them to output directory.
        """
        if not self.output_dir:
            raise ValueError("Output directory required for report generation")
            
        # 1. Category distributions
        self.plot_category_distributions(scores)
        
        # 2. Effect size heatmaps
        d_matrix = np.zeros((len(results.categories), len(results.categories)))
        cles_matrix = np.zeros_like(d_matrix)
        
        for i, cat1 in enumerate(results.categories):
            for j, cat2 in enumerate(results.categories):
                if (cat1, cat2) in results.effect_sizes:
                    comparison = results.effect_sizes[(cat1, cat2)]
                    d_matrix[i,j] = comparison.cohens_d
                    cles_matrix[i,j] = comparison.cles
                    
        self.plot_effect_size_heatmap(
            d_matrix,
            results.categories,
            "Cohen's d Effect Sizes"
        )
        
        self.plot_effect_size_heatmap(
            cles_matrix,
            results.categories,
            "Common Language Effect Sizes"
        )
        
        # 3. Bootstrap distributions if available
        if bootstrap_data is not None:
            self.plot_bootstrap_distributions(bootstrap_data)
            
        logger.info(f"Analysis report generated in {self.output_dir}")