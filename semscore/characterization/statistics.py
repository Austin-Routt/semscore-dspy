"""
Statistical analysis module for SemScore characterization.
Provides functions for computing various statistical measures and effect sizes
between semantic similarity categories.

This module implements:
1. Descriptive statistics with confidence intervals
2. Effect size calculations (Cohen's d, CLES)
3. Bootstrapping analysis
4. Distribution comparison statistics
5. Pairwise statistical tests
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class StatisticalSummary:
    """Container for statistical analysis results."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    median: float
    iqr: float
    n: int
    skewness: Optional[float]  # Optional for small samples
    kurtosis: Optional[float]  # Optional for small samples
    normality_pvalue: Optional[float]  # Optional for small samples

def calculate_cohens_d(group1: np.ndarray, 
                      group2: np.ndarray,
                      bias_correction: bool = True) -> float:
    """
    Calculate Cohen's d effect size with optional bias correction.
    
    Args:
        group1: First group's scores
        group2: Second group's scores
        bias_correction: Whether to apply Hedges' correction for small samples
        
    Returns:
        float: Cohen's d effect size value (absolute value)
        
    Notes:
        - Uses pooled standard deviation
        - Applies Hedges' correction when bias_correction=True
        - Handles unequal sample sizes
        - Returns 0 if both groups have zero variance
        - Returns absolute value for consistent comparison
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_sd == 0:
        logger.warning("Both groups have zero variance, returning d = 0")
        return 0.0
        
    # Calculate basic Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_sd
    
    if bias_correction and (n1 + n2 - 2) > 0:
        # Apply Hedges' correction for small sample bias
        correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        d = d * correction
        
    # Return absolute value for consistent comparison
    return float(abs(d))

def calculate_cles(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Common Language Effect Size (CLES).
    
    CLES represents the probability that a randomly selected value from
    group1 will be greater than a randomly selected value from group2.
    For tied values, each outcome (greater or less) is given 0.5 probability.
    
    Args:
        group1: First group's scores
        group2: Second group's scores
        
    Returns:
        float: CLES value between 0 and 1
        
    Notes:
        - 0.5 indicates no effect (random chance)
        - Values closer to 1 indicate group1 tends to be larger
        - Values closer to 0 indicate group2 tends to be larger
        - Ties are handled by adding half their count
    """
    x = group1[:, np.newaxis]
    y = group2[np.newaxis, :]
    
    # Count strict inequalities and ties
    greater = np.sum(x > y)
    ties = np.sum(x == y)
    total = len(group1) * len(group2)
    
    # Add half of ties to greater count
    cles = (greater + 0.5 * ties) / total
    
    return float(cles)

def bootstrap_sample(data: np.ndarray,
                    statistic: callable,
                    n_replicates: int = 10000,
                    random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate bootstrap samples and compute statistics.
    """
    if len(data) == 0:
        raise ValueError("No valid data points for bootstrap")
        
    if random_state is not None:
        np.random.seed(random_state)
        
    # Remove NaN values if present
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        raise ValueError("Input data contains no valid values")
        
    # Generate bootstrap samples with replacement
    indices = np.random.randint(0, len(clean_data), 
                              size=(n_replicates, len(clean_data)))
    samples = clean_data[indices]
    
    # Apply statistic to each sample
    try:
        bootstrap_stats = np.array([statistic(sample) for sample in samples])
    except Exception as e:
        logger.error(f"Error computing bootstrap statistics: {str(e)}")
        raise
        
    return bootstrap_stats

def calculate_ci(data: np.ndarray,
                confidence: float = 0.95,
                method: str = 'percentile') -> Tuple[float, float]:
    """
    Calculate confidence intervals using various methods.
    
    Args:
        data: Input data array
        confidence: Confidence level (between 0 and 1)
        method: Method for CI calculation ('percentile' or 'basic')
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = (1 - confidence) / 2
    
    if method == 'percentile':
        # Using np.percentile directly to avoid scipy warning
        ci_lower = np.percentile(data, 100 * alpha)
        ci_upper = np.percentile(data, 100 * (1 - alpha))
    else:  # basic bootstrap
        theta_hat = np.mean(data)
        z_alpha = stats.norm.ppf(alpha)
        z_1_alpha = stats.norm.ppf(1 - alpha)
        
        se = np.std(data, ddof=1) / np.sqrt(len(data))
        ci_lower = theta_hat - z_1_alpha * se
        ci_upper = theta_hat - z_alpha * se
        
    return float(ci_lower), float(ci_upper)

def analyze_distribution(data: np.ndarray) -> StatisticalSummary:
    """
    Compute comprehensive statistical summary of distribution.
    """
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 2:
        raise ValueError("Need at least 2 valid data points for analysis")
        
    # Basic statistics
    mean = np.mean(clean_data)
    std = np.std(clean_data, ddof=1)
    median = np.median(clean_data)
    iqr = stats.iqr(clean_data)
    
    # Calculate confidence intervals using percentile method
    # as it's more robust for small samples
    ci_lower, ci_upper = calculate_ci(clean_data, method='percentile')
    
    # Shape statistics - only if enough samples
    if len(clean_data) >= 8:
        skewness = float(stats.skew(clean_data))
        kurtosis = float(stats.kurtosis(clean_data))
        _, normality_pvalue = stats.normaltest(clean_data)
        normality_pvalue = float(normality_pvalue)
    else:
        skewness = None
        kurtosis = None
        normality_pvalue = None
    
    return StatisticalSummary(
        mean=float(mean),
        std=float(std),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median=float(median),
        iqr=float(iqr),
        n=len(clean_data),
        skewness=skewness,
        kurtosis=kurtosis,
        normality_pvalue=normality_pvalue
    )