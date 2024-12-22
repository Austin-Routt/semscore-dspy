"""
Visualization utilities for SemScore metric analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SemScorePlotter:
    """Handles visualization of SemScore distributions and comparisons."""
    
    def __init__(self, output_dir: Union[str, Path] = "semscore_results"):
        """
        Initialize plotter with output directory.
        
        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized plotter with output directory: {self.output_dir}")

    def plot_score_distribution(
        self,
        scores: Union[List[float], np.ndarray],
        name: str,
        timestamp: str,
        save: bool = True,
        show: bool = False,
        **kwargs
    ) -> Optional[Path]:
        """
        Create and optionally save distribution plot for scores.
        
        Args:
            scores: List of similarity scores
            name: Name/identifier for the plot
            timestamp: Timestamp for file naming
            save: Whether to save the plot
            show: Whether to display the plot
            **kwargs: Additional kwargs for seaborn histplot
        
        Returns:
            Path to saved plot if save=True, else None
        """
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(
            np.array(scores),
            kde=True,
            **{
                'bins': 50,
                'color': 'blue',
                'alpha': 0.6,
                **kwargs
            }
        )
        
        # Customize plot
        plt.title(f'Score Distribution for {name}')
        plt.xlabel('Semantic Similarity Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistical annotations
        stats_text = (
            f'Mean: {np.mean(scores):.3f}\n'
            f'Median: {np.median(scores):.3f}\n'
            f'Std: {np.std(scores):.3f}'
        )
        plt.text(
            0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        if save:
            plot_path = self.output_dir / f'score_dist_{name.replace("/", "_")}_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return plot_path if save else None

    def plot_comparison(
        self,
        calculated_scores: List[float],
        expected_scores: List[float],
        name: str,
        timestamp: str,
        save: bool = True,
        show: bool = False
    ) -> Optional[Path]:
        """
        Create scatter plot comparing calculated vs expected scores.
        
        Args:
            calculated_scores: Scores from current implementation
            expected_scores: Reference scores
            name: Name/identifier for the plot
            timestamp: Timestamp for file naming
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Path to saved plot if save=True, else None
        """
        plt.figure(figsize=(8, 8))
        
        # Create scatter plot
        plt.scatter(expected_scores, calculated_scores, alpha=0.5)
        
        # Add diagonal line for perfect correlation
        lims = [
            min(plt.xlim()[0], plt.ylim()[0]),
            max(plt.xlim()[1], plt.ylim()[1])
        ]
        plt.plot(lims, lims, 'r--', alpha=0.8, label='Perfect correlation')
        
        # Customize plot
        plt.title(f'Calculated vs Expected Scores for {name}')
        plt.xlabel('Expected Scores')
        plt.ylabel('Calculated Scores')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add correlation annotation
        corr = np.corrcoef(expected_scores, calculated_scores)[0, 1]
        plt.text(
            0.05, 0.95,
            f'Correlation: {corr:.3f}',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        if save:
            plot_path = self.output_dir / f'comparison_{name.replace("/", "_")}_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return plot_path if save else None