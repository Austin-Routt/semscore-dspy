"""
Test suite for SemScore visualization module.
Tests plotting functionality and visualization output.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import shutil
import logging

from semscore.characterization.visualization import SemScoreVisualizer
from semscore.characterization.distributions import (
    CategoryDistributions,
    DistributionComparison
)

logger = logging.getLogger(__name__)

class TestVisualization(unittest.TestCase):
    """Test cases for visualization functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and temporary directory."""
        # Create temporary directory for test outputs
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create test data
        np.random.seed(42)
        
        # Generate realistic test distributions
        def generate_test_dist(mean: float, std: float, size: int) -> np.ndarray:
            """Generate controlled test distribution."""
            base = np.random.normal(mean, std, size)
            noise = np.random.uniform(-0.02, 0.02, size)
            return np.clip(base + noise, 0, 1)
        
        cls.test_data = {
            "Identical": generate_test_dist(0.95, 0.02, 50),
            "Paraphrase": generate_test_dist(0.85, 0.05, 50),
            "Different": generate_test_dist(0.30, 0.10, 50)
        }
        
        # Generate bootstrap data
        cls.bootstrap_data = {
            cat: generate_test_dist(np.mean(data), np.std(data), 1000)
            for cat, data in cls.test_data.items()
        }
        
        # Create mock CategoryDistributions for testing
        cls.mock_results = CategoryDistributions(
            categories=list(cls.test_data.keys()),
            means={cat: float(np.mean(data)) 
                  for cat, data in cls.test_data.items()},
            stds={cat: float(np.std(data)) 
                  for cat, data in cls.test_data.items()},
            medians={cat: float(np.median(data)) 
                    for cat, data in cls.test_data.items()},
            effect_sizes={(cat1, cat2): DistributionComparison(
                cohens_d=1.0,
                cles=0.8,
                ks_statistic=0.5,
                ks_pvalue=0.01,
                mean_diff=0.2,
                ci_diff_lower=0.1,
                ci_diff_upper=0.3
            ) for cat1 in cls.test_data for cat2 in cls.test_data if cat1 != cat2},
            normality_tests={cat: (0.5, 0.5) 
                           for cat in cls.test_data},
            sample_sizes={cat: len(data) 
                        for cat, data in cls.test_data.items()}
        )
        
        # Initialize visualizer
        cls.visualizer = SemScoreVisualizer(
            output_dir=str(cls.test_dir)
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.test_dir)

    def test_initialization(self):
        """Test proper initialization of visualizer."""
        self.assertTrue(self.test_dir.exists())
        self.assertIsNotNone(self.visualizer.colors)
        self.assertIsNotNone(self.visualizer.cmap)

    def test_category_distributions(self):
        """Test category distribution plotting."""
        # Test with default settings
        fig = self.visualizer.plot_category_distributions(
            self.test_data
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test violin plot
        fig = self.visualizer.plot_category_distributions(
            self.test_data,
            violin=True
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Verify file creation
        plot_file = self.test_dir / "category_distributions.png"
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)

    def test_effect_size_heatmap(self):
        """Test effect size heatmap plotting."""
        categories = list(self.test_data.keys())
        n_cats = len(categories)
        
        # Create test matrices
        effect_matrix = np.random.rand(n_cats, n_cats)
        np.fill_diagonal(effect_matrix, 0)
        
        # Test default settings
        fig = self.visualizer.plot_effect_size_heatmap(
            effect_matrix,
            categories
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with annotations off
        fig = self.visualizer.plot_effect_size_heatmap(
            effect_matrix,
            categories,
            annotate=False
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Verify file creation
        plot_file = self.test_dir / "effect_size_heatmap.png"
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)

    def test_bootstrap_distributions(self):
        """Test bootstrap distribution plotting."""
        # Test with default settings
        fig = self.visualizer.plot_bootstrap_distributions(
            self.bootstrap_data
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test without individual plots
        fig = self.visualizer.plot_bootstrap_distributions(
            self.bootstrap_data,
            show_individual=False
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Verify file creation
        plot_file = self.test_dir / "bootstrap_distributions.png"
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)

    def test_analysis_report(self):
        """Test complete analysis report generation."""
        # Test without bootstrap data
        self.visualizer.create_analysis_report(
            self.mock_results,
            self.test_data
        )
        
        # Test with bootstrap data
        self.visualizer.create_analysis_report(
            self.mock_results,
            self.test_data,
            self.bootstrap_data
        )
        
        # Verify all expected files exist
        expected_files = [
            "category_distributions.png",
            "effect_size_heatmap.png",
            "bootstrap_distributions.png"
        ]
        
        for filename in expected_files:
            filepath = self.test_dir / filename
            self.assertTrue(filepath.exists())
            self.assertGreater(filepath.stat().st_size, 0)

    def test_error_handling(self):
        """Test error handling in visualization."""
        # Test with invalid output directory
        with self.assertRaises(ValueError):
            visualizer = SemScoreVisualizer(output_dir=None)
            visualizer.create_analysis_report(
                self.mock_results,
                self.test_data
            )
        
        # Test with empty data
        empty_data = {}
        with self.assertRaises(ValueError):
            self.visualizer.plot_category_distributions(empty_data)

def main():
    """Run the test suite."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()