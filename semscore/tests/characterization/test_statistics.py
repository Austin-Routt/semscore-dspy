"""
Test suite for SemScore statistical analysis module.
Tests all statistical functions with known distributions and edge cases.
"""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from typing import List, Tuple
import logging

from semscore.characterization.statistics import (
    calculate_cohens_d,
    calculate_cles,
    bootstrap_sample,
    calculate_ci,
    analyze_distribution,
    StatisticalSummary
)

logger = logging.getLogger(__name__)

class TestStatistics(unittest.TestCase):
    """Test cases for statistical analysis functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across tests."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create controlled test distributions
        cls.dist1 = np.array([-1, -0.5, 0, 0.5, 1])  # mean=0, std≈0.91
        cls.dist2 = np.array([0, 0.5, 1, 1.5, 2])    # mean=1, std≈0.91
        
        # Large sample distributions
        cls.large_dist1 = np.concatenate([cls.dist1] * 200)  # n=1000
        cls.large_dist2 = np.concatenate([cls.dist2] * 200)  # n=1000
        
        # Edge cases
        cls.single_value = np.array([1.0])
        cls.identical_values = np.array([1.0] * 10)
        cls.small_sample = np.array([0, 0.1, 0.2, 0.3, 0.4])

    def test_cohens_d(self):
        """Test Cohen's d calculation with known distributions."""
        # Test with controlled distributions
        d = calculate_cohens_d(self.dist1, self.dist2)
        self.assertAlmostEqual(d, 1.1, places=1)  # Known effect size
        
        # Test with identical distributions
        d_zero = calculate_cohens_d(self.dist1, self.dist1)
        self.assertAlmostEqual(d_zero, 0, places=2)
        
        # Test with constant values
        d_const = calculate_cohens_d(self.identical_values, self.identical_values)
        self.assertEqual(d_const, 0)

    def test_cles(self):
        """Test Common Language Effect Size calculation."""
        # Test with controlled distributions
        cles = calculate_cles(self.dist2, self.dist1)
        self.assertGreater(cles, 0.5)
        
        # Test with identical distributions
        cles_same = calculate_cles(self.dist1, self.dist1)
        self.assertAlmostEqual(cles_same, 0.5, places=2)
        
        # Test with completely separated distributions
        high = np.array([10, 11, 12])
        low = np.array([1, 2, 3])
        cles_separated = calculate_cles(high, low)
        self.assertEqual(cles_separated, 1.0)
        
        # Additional test for ties
        tied = np.array([1, 1, 1])
        cles_tied = calculate_cles(tied, tied)
        self.assertEqual(cles_tied, 0.5)

    def test_bootstrap(self):
        """Test bootstrap sampling and statistical computation."""
        def mean_statistic(x):
            return np.mean(x)
            
        # Test with controlled distribution
        bootstrap_means = bootstrap_sample(self.dist1, mean_statistic, 
                                        n_replicates=1000,
                                        random_state=42)
        self.assertEqual(len(bootstrap_means), 1000)
        self.assertAlmostEqual(np.mean(bootstrap_means), 0, places=1)
        
        # Test reproducibility
        means1 = bootstrap_sample(self.dist1, mean_statistic, 
                                n_replicates=100, random_state=42)
        means2 = bootstrap_sample(self.dist1, mean_statistic,
                                n_replicates=100, random_state=42)
        np.testing.assert_array_equal(means1, means2)
        
        # Test with empty input
        with self.assertRaises(ValueError):
            bootstrap_sample(np.array([]), mean_statistic)

    def test_confidence_intervals(self):
        """Test confidence interval calculations."""
        # Test with controlled distribution
        lower, upper = calculate_ci(self.dist1)
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)
        
        # Test different confidence levels
        l90, u90 = calculate_ci(self.dist1, confidence=0.90)
        l95, u95 = calculate_ci(self.dist1, confidence=0.95)
        self.assertGreater(u95 - l95, u90 - l90)
        
    def test_distribution_analysis(self):
        """Test comprehensive distribution analysis."""
        # Test with small sample (n < 8)
        stats_small = analyze_distribution(self.dist1)
        self.assertIsInstance(stats_small, StatisticalSummary)
        self.assertIsNone(stats_small.skewness)
        self.assertIsNone(stats_small.kurtosis)
        self.assertIsNone(stats_small.normality_pvalue)
        
        # Test with large sample
        stats_large = analyze_distribution(self.large_dist1)
        self.assertIsNotNone(stats_large.skewness)
        self.assertIsNotNone(stats_large.kurtosis)
        self.assertIsNotNone(stats_large.normality_pvalue)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Data with NaN values
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        stats = analyze_distribution(data_with_nan)
        self.assertEqual(stats.n, 4)
        
        # Too few samples
        with self.assertRaises(ValueError):
            analyze_distribution(self.single_value)
            
        # Equal values
        stats_equal = analyze_distribution(self.identical_values)
        self.assertEqual(stats_equal.std, 0)
        
def main():
    """Run the test suite."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()