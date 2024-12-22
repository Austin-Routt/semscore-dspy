"""
Test suite for distribution analysis module.
Tests distribution comparison, bootstrapping, and effect size calculations.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from semscore.characterization.distributions import (
    analyze_category_distributions,
    generate_bootstrap_distributions,
    validate_distributions,
    calculate_effect_matrices,
    CategoryDistributions,
    DistributionComparison
)

logger = logging.getLogger(__name__)

class TestDistributions(unittest.TestCase):
    """Test cases for distribution analysis functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all test methods."""
        np.random.seed(42)  # For reproducibility
        
        # Create realistic test distributions with overlap
        cls.test_data = {
            "Identical": np.array([0.90, 0.91, 0.89, 0.92, 0.88]),
            "Paraphrase": np.array([0.85, 0.87, 0.82, 0.88, 0.84]),
            "Different": np.array([0.25, 0.28, 0.22, 0.30, 0.26])
        }
        
        # Create larger test distributions with controlled overlap
        def generate_overlapping_dist(mean: float, std: float, size: int) -> np.ndarray:
            """Generate normally distributed data with clipping and noise."""
            base = np.random.normal(mean, std, size)
            noise = np.random.uniform(-0.02, 0.02, size)  # Add small random noise
            return np.clip(base + noise, 0, 1)
        
        cls.large_test_data = {
            "Identical": generate_overlapping_dist(0.90, 0.03, 100),
            "Paraphrase": generate_overlapping_dist(0.80, 0.05, 100),
            "Different": generate_overlapping_dist(0.30, 0.08, 100)
        }

    def test_category_analysis(self):
        """Test analysis of category distributions."""
        results = analyze_category_distributions(self.test_data)
        
        # Check basic structure
        self.assertIsInstance(results, CategoryDistributions)
        self.assertEqual(set(results.categories),
                        set(self.test_data.keys()))
                        
        # Check means are in valid range and properly ordered
        self.assertGreater(results.means["Identical"], results.means["Paraphrase"])
        self.assertGreater(results.means["Paraphrase"], results.means["Different"])
        
        # Check effect sizes
        for (cat1, cat2), comparison in results.effect_sizes.items():
            self.assertIsInstance(comparison, DistributionComparison)
            self.assertGreaterEqual(comparison.cles, 0)
            self.assertLessEqual(comparison.cles, 1)
            
            # Verify confidence intervals make sense
            self.assertLess(comparison.ci_diff_lower, comparison.ci_diff_upper)
            self.assertGreaterEqual(comparison.mean_diff, comparison.ci_diff_lower)
            self.assertLessEqual(comparison.mean_diff, comparison.ci_diff_upper)

    def test_bootstrap_distributions(self):
        """Test bootstrap distribution generation."""
        n_bootstrap = 1000
        bootstrap_results = generate_bootstrap_distributions(
            self.test_data,
            n_bootstrap=n_bootstrap,
            random_state=42
        )
        
        # Check sizes and basic properties
        for cat in self.test_data:
            dist = bootstrap_results[cat]
            self.assertEqual(len(dist), n_bootstrap)
            self.assertTrue(np.all(np.isfinite(dist)))
            self.assertTrue(np.all((dist >= 0) & (dist <= 1)))
            
            # Verify mean is preserved within tolerance
            orig_mean = np.mean(self.test_data[cat])
            boot_mean = np.mean(dist)
            self.assertAlmostEqual(orig_mean, boot_mean, places=2)
            
        # Verify bootstrap preserves ordering
        id_means = np.mean(bootstrap_results["Identical"], axis=0)
        para_means = np.mean(bootstrap_results["Paraphrase"], axis=0)
        diff_means = np.mean(bootstrap_results["Different"], axis=0)
        
        self.assertGreater(np.mean(id_means), np.mean(para_means))
        self.assertGreater(np.mean(para_means), np.mean(diff_means))

    def test_effect_matrices(self):
        """Test effect size matrix calculations."""
        d_matrix, cles_matrix = calculate_effect_matrices(self.test_data)
        
        # Check matrix properties
        categories = list(self.test_data.keys())
        self.assertEqual(list(d_matrix.index), categories)
        self.assertEqual(list(cles_matrix.columns), categories)
        
        # Check diagonal values
        for cat in categories:
            self.assertEqual(d_matrix.loc[cat, cat], 0.0)
            self.assertAlmostEqual(cles_matrix.loc[cat, cat], 0.5)
        
        # Verify expected relationships with tolerance
        def check_relationship(high_cat: str, mid_cat: str, low_cat: str):
            """Helper to check category relationships."""
            high_low_cles = cles_matrix.loc[high_cat, low_cat]
            mid_low_cles = cles_matrix.loc[mid_cat, low_cat]
            
            # Allow small tolerance for floating point
            self.assertGreaterEqual(high_low_cles, mid_low_cles - 1e-10)
            
        check_relationship("Identical", "Paraphrase", "Different")

    def test_large_scale_analysis(self):
        """Test analysis with larger datasets."""
        results = analyze_category_distributions(self.large_test_data)
        
        # Test discrimination between categories
        id_diff = results.effect_sizes[("Identical", "Different")]
        para_diff = results.effect_sizes[("Paraphrase", "Different")]
        
        # Use assertGreaterEqual with small tolerance for floating point
        self.assertGreaterEqual(id_diff.cohens_d, para_diff.cohens_d - 1e-10)
        self.assertGreaterEqual(id_diff.cles, para_diff.cles - 1e-10)
        
        # Test normality results
        for cat in self.large_test_data:
            stat, pval = results.normality_tests[cat]
            self.assertTrue(np.isfinite(stat))
            self.assertTrue(np.isfinite(pval))
            self.assertGreaterEqual(pval, 0)
            self.assertLessEqual(pval, 1)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty data
        empty_data = {"Empty": np.array([])}
        validation = validate_distributions(empty_data)
        self.assertGreater(len(validation["Empty"]), 0)
        
        # Single value
        single_data = {"Single": np.array([0.5])}
        validation = validate_distributions(single_data)
        self.assertGreater(len(validation["Single"]), 0)
        
        # Invalid values
        invalid_data = {"Invalid": np.array([0.5, np.nan, 2.0, -1.0])}
        validation = validate_distributions(invalid_data)
        self.assertTrue(any("NaN" in msg for msg in validation["Invalid"]))
        self.assertTrue(any("outside valid range" in msg for msg in validation["Invalid"]))
        
        # Check specific validation message content
        messages = validation["Invalid"]
        range_messages = [msg for msg in messages if "outside valid range" in msg]
        self.assertEqual(len(range_messages), 1)
        msg = range_messages[0]
        self.assertIn("min=-1.00", msg)
        self.assertIn("max=2.00", msg)

def main():
    """Run the test suite."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()