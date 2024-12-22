"""
Distribution analysis module for SemScore characterization.
Provides functions for analyzing and comparing distributions of semantic similarity scores.

This module implements:
1. Distribution comparison tests
2. Effect size matrices
3. Multi-group analysis
4. Distribution validation and quality checks
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import pandas as pd
from .statistics import calculate_cohens_d, calculate_ci

logger = logging.getLogger(__name__)

@dataclass
class DistributionComparison:
    """Results container for distribution comparison analysis."""
    cohens_d: float
    cles: float
    ks_statistic: float
    ks_pvalue: float
    mean_diff: float
    ci_diff_lower: float
    ci_diff_upper: float

@dataclass
class CategoryDistributions:
    """Container for semantic score distributions by category."""
    categories: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    medians: Dict[str, float]
    effect_sizes: Dict[Tuple[str, str], DistributionComparison]
    normality_tests: Dict[str, Tuple[float, float]]
    sample_sizes: Dict[str, int]

def analyze_category_distributions(
    scores: Dict[str, np.ndarray],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> CategoryDistributions:
    """
    Analyze distributions of semantic scores across categories.
    
    Args:
        scores: Dictionary mapping categories to their score arrays
        confidence: Confidence level for intervals (default: 0.95)
        n_bootstrap: Number of bootstrap samples (default: 10000)
        
    Returns:
        CategoryDistributions object containing analysis results
    """
    categories = list(scores.keys())
    means = {}
    stds = {}
    medians = {}
    effect_sizes = {}
    normality_tests = {}
    sample_sizes = {}
    
    # Calculate basic statistics for each category
    for category in categories:
        data = scores[category]
        means[category] = float(np.mean(data))
        stds[category] = float(np.std(data, ddof=1))
        medians[category] = float(np.median(data))
        sample_sizes[category] = len(data)
        
        # Normality test only if enough samples
        if len(data) >= 8:
            statistic, pvalue = stats.normaltest(data)
            normality_tests[category] = (float(statistic), float(pvalue))
        else:
            normality_tests[category] = (np.nan, np.nan)
            logger.warning(f"Category {category} has insufficient samples for normality testing")
    
    # Calculate pairwise effect sizes and comparisons
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories[i+1:], i+1):
            data1 = scores[cat1]
            data2 = scores[cat2]
            
            # Calculate Cohen's d
            d = calculate_cohens_d(data1, data2)
            
            # Calculate CLES P(X>Y)
            x = data1[:, np.newaxis]
            y = data2[np.newaxis, :]
            cles = np.mean(x > y) + 0.5 * np.mean(x == y)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(data1, data2)
            
            # Calculate mean difference and CI
            mean_diff = np.mean(data1) - np.mean(data2)
            
            # Bootstrap confidence interval for mean difference
            diffs = []
            for _ in range(n_bootstrap):
                boot1 = np.random.choice(data1, size=len(data1), replace=True)
                boot2 = np.random.choice(data2, size=len(data2), replace=True)
                diffs.append(np.mean(boot1) - np.mean(boot2))
                
            ci_lower, ci_upper = calculate_ci(
                np.array(diffs),
                confidence=confidence
            )
            
            # Store comparison results for both directions
            comparison = DistributionComparison(
                cohens_d=d,
                cles=cles,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                mean_diff=float(mean_diff),
                ci_diff_lower=float(ci_lower),
                ci_diff_upper=float(ci_upper)
            )
            
            effect_sizes[(cat1, cat2)] = comparison
            # Store complementary comparison
            effect_sizes[(cat2, cat1)] = DistributionComparison(
                cohens_d=d,
                cles=1-cles,  # Complement for opposite direction
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                mean_diff=float(-mean_diff),  # Negate for opposite direction
                ci_diff_lower=float(-ci_upper),  # Swap and negate CI bounds
                ci_diff_upper=float(-ci_lower)
            )
    
    return CategoryDistributions(
        categories=categories,
        means=means,
        stds=stds,
        medians=medians,
        effect_sizes=effect_sizes,
        normality_tests=normality_tests,
        sample_sizes=sample_sizes
    )

def generate_bootstrap_distributions(
    scores: Dict[str, np.ndarray],
    n_bootstrap: int = 10000,
    statistic: callable = np.mean,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate bootstrap distributions for each category.
    
    Args:
        scores: Dictionary mapping categories to their score arrays
        n_bootstrap: Number of bootstrap samples
        statistic: Function to compute on bootstrap samples
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping categories to their bootstrap distributions
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    bootstrap_dists = {}
    
    for category, data in scores.items():
        if len(data) == 0:
            logger.warning(f"Skipping empty category: {category}")
            continue
            
        # Generate bootstrap samples
        indices = np.random.randint(0, len(data), 
                                  size=(n_bootstrap, len(data)))
        samples = data[indices]
        
        # Calculate statistic for each sample
        try:
            bootstrap_dists[category] = np.array([
                statistic(sample) for sample in samples
            ])
        except Exception as e:
            logger.error(f"Error calculating bootstrap for {category}: {str(e)}")
            raise
            
    return bootstrap_dists

def validate_distributions(
    scores: Dict[str, np.ndarray],
    min_samples: int = 8,
    check_normality: bool = True
) -> Dict[str, List[str]]:
    """
    Validate score distributions for analysis requirements.
    
    Args:
        scores: Dictionary mapping categories to their score arrays
        min_samples: Minimum required samples per category
        check_normality: Whether to check for normality
        
    Returns:
        Dictionary mapping categories to list of validation messages
    """
    validation_results = {cat: [] for cat in scores.keys()}
    
    for category, data in scores.items():
        # Check sample size
        if len(data) < min_samples:
            validation_results[category].append(
                f"Insufficient samples: {len(data)} < {min_samples}"
            )
        
        # Remove NaN values for valid range checking
        valid_data = data[~np.isnan(data)]
        
        # Check for invalid values
        n_nan = np.sum(np.isnan(data))
        if n_nan > 0:
            validation_results[category].append(
                f"Contains {n_nan} NaN values"
            )
            
        n_inf = np.sum(~np.isfinite(data))
        if n_inf > 0:
            validation_results[category].append(
                f"Contains {n_inf} infinite values"
            )
            
        # Check score range on valid data
        if len(valid_data) > 0:
            invalid_range = ((valid_data < 0) | (valid_data > 1))
            n_invalid = np.sum(invalid_range)
            if n_invalid > 0:
                min_val = np.min(valid_data[invalid_range])
                max_val = np.max(valid_data[invalid_range])
                validation_results[category].append(
                    f"Contains {n_invalid} scores outside valid range [0,1]: min={min_val:.2f}, max={max_val:.2f}"
                )
                
        # Check normality if requested and enough samples
        if check_normality and len(valid_data) >= min_samples:
            try:
                _, p_value = stats.normaltest(valid_data)
                if p_value < 0.05:
                    validation_results[category].append(
                        f"Non-normal distribution (p={p_value:.4f})"
                    )
            except Exception as e:
                validation_results[category].append(
                    f"Error testing normality: {str(e)}"
                )
                
    return validation_results


def calculate_effect_matrices(data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate effect size matrices for all category pairs.
    
    Args:
        data: Either a DataFrame with 'category' and 'semantic_score' columns,
              or a dictionary mapping categories to score arrays
        
    Returns:
        Tuple of (cohens_d_matrix, cles_matrix) as pandas DataFrames
    """
    # Handle input type
    if isinstance(data, pd.DataFrame):
        categories = data['category'].unique()
        scores_dict = {cat: data[data['category'] == cat]['semantic_score'].values 
                      for cat in categories}
    else:
        categories = list(data.keys())
        scores_dict = data
    
    # Initialize matrices
    d_matrix = pd.DataFrame(0.0, 
                          index=categories,
                          columns=categories)
    cles_matrix = pd.DataFrame(0.5,
                             index=categories,
                             columns=categories)
                             
    # Calculate effect sizes
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i != j:  # Skip diagonal
                data1 = scores_dict[cat1]
                data2 = scores_dict[cat2]
                
                # Calculate Cohen's d
                d = calculate_cohens_d(data1, data2)
                d_matrix.iloc[i,j] = d
                
                # Calculate CLES (P(X>Y))
                x = data1[:, np.newaxis]
                y = data2[np.newaxis, :]
                cles = np.mean(x > y) + 0.5 * np.mean(x == y)
                cles_matrix.iloc[i,j] = cles
                
                # Set complementary CLES value
                cles_matrix.iloc[j,i] = 1 - cles
    
    return d_matrix, cles_matrix