"""Core implementation of SemScore characterization suite."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from jsonschema import validate, ValidationError

import numpy as np
import pandas as pd

from ..metric.core import SemScoreMetric
from ..utils.validation import SemScoreValidator

logger = logging.getLogger(__name__)

class CharacterizationSuite:
    """Characterization suite for evaluating SemScore metric behavior."""
    
    def __init__(
        self,
        schema_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        metric: Optional[SemScoreMetric] = None  # Add metric parameter
    ):
        """Initialize characterization suite.
        
        Args:
            schema_path: Path to schema JSON file
            dataset_path: Path to dataset JSON file
            output_dir: Directory for output files
        """
        # Set default paths relative to module
        self.base_path = Path(__file__).parent.parent
        self.schema_path = Path(schema_path or self.base_path / "data/characterization/SemScore_Characterization_Schema.json")
        self.dataset_path = Path(dataset_path or self.base_path / "data/characterization/SemScore_Characterization_Dataset.json")
        self.output_dir = Path(output_dir or "characterization_results")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or use provided metric
        self.metric = metric or SemScoreMetric()
        
        # Load and validate dataset
        self.schema = self._load_json(self.schema_path)
        self.dataset = self._load_json(self.dataset_path)
        self._validate_dataset()
        
        # Convert to DataFrame for analysis
        self.df = pd.DataFrame(self.dataset)

    def _load_json(self, filepath: Path) -> Dict:
        """Load and parse JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            raise

    def _validate_dataset(self) -> None:
        """Validate dataset against schema with detailed error messages."""
        validation_errors = []
        
        for idx, item in enumerate(self.dataset):
            try:
                # Validate required fields first
                required_fields = ['category', 'domain', 'text1', 'text2', 'expected_range']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    validation_errors.append(
                        f"Item {idx}: Missing required fields: {', '.join(missing_fields)}"
                    )
                    continue
                    
                # Validate string lengths
                for field in ['text1', 'text2']:
                    if not item[field].strip():
                        validation_errors.append(
                            f"Item {idx}: Field '{field}' cannot be empty"
                        )
                        
                # Validate expected_range
                expected_range = item['expected_range']
                if not isinstance(expected_range, list) or len(expected_range) != 2:
                    validation_errors.append(
                        f"Item {idx}: expected_range must be a list of 2 numbers"
                    )
                elif not all(isinstance(x, (int, float)) for x in expected_range):
                    validation_errors.append(
                        f"Item {idx}: expected_range values must be numbers"
                    )
                elif not all(0 <= x <= 1 for x in expected_range):
                    validation_errors.append(
                        f"Item {idx}: expected_range values must be between 0 and 1"
                    )
                    
                # Full schema validation
                validate(instance=item, schema=self.schema)
                
            except ValidationError as e:
                validation_errors.append(f"Item {idx}: {e.message}")
                
        if validation_errors:
            error_msg = "\n".join(validation_errors)
            logger.error(f"Dataset validation failed:\n{error_msg}")
            raise ValidationError(error_msg)

    def run_characterization(self) -> Dict[str, Any]:
        """Run complete characterization analysis."""
        logger.info("Starting characterization analysis...")
        
        results = {
            'category_analysis': self._analyze_by_category(),
            'domain_analysis': self._analyze_by_domain(),
            'range_compliance': self._analyze_range_compliance(),
            'overall_stats': self._calculate_overall_stats()
        }
        
        # Save results
        self._save_results(results)
        
        return results
    

    def _check_range_compliance(self, scores: List[float], expected_ranges: List[List[float]]) -> Dict[str, Any]:
        """Check if scores fall within expected ranges."""
        compliant = 0
        total = len(scores)
        
        for score, expected_range in zip(scores, expected_ranges):
            if expected_range[0] <= score <= expected_range[1]:
                compliant += 1
                
        return {
            'compliant_count': compliant,
            'total_count': total,
            'compliance_rate': float(compliant / total) if total > 0 else 0.0
        }

    def _calculate_domain_category_stats(self, domain_data: pd.DataFrame, scores: List[float]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each category within a domain."""
        stats = {}
        
        for category in domain_data['category'].unique():
            cat_mask = domain_data['category'] == category
            cat_scores = [s for s, m in zip(scores, cat_mask) if m]
            
            if cat_scores:
                stats[category] = {
                    'mean': float(np.mean(cat_scores)),
                    'std': float(np.std(cat_scores)) if len(cat_scores) > 1 else 0.0,
                    'count': len(cat_scores)
                }
                
        return stats

    def _analyze_by_category(self) -> Dict[str, Dict]:
        """Analyze metric performance by semantic category."""
        categories = self.df['category'].unique()
        results = {}
        
        for category in categories:
            cat_data = self.df[self.df['category'] == category]
            scores = []
            
            for _, row in cat_data.iterrows():
                score = self.metric(row['text1'], row['text2'])
                scores.append(score)
            
            results[category] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'median_score': float(np.median(scores)),
                'range_compliance': self._check_range_compliance(
                    scores, 
                    cat_data['expected_range'].tolist()
                ),
                'sample_count': len(scores)
            }
        
        return results

    def _analyze_by_domain(self) -> Dict[str, Dict]:
        """Analyze metric performance by domain."""
        domains = self.df['domain'].unique()
        results = {}
        
        for domain in domains:
            domain_data = self.df[self.df['domain'] == domain]
            scores = []
            
            for _, row in domain_data.iterrows():
                score = self.metric(row['text1'], row['text2'])
                scores.append(score)
            
            results[domain] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'score_by_category': self._calculate_domain_category_stats(
                    domain_data, scores
                ),
                'sample_count': len(scores)
            }
        
        return results

    def _analyze_range_compliance(self) -> Dict[str, float]:
        """Analyze how well scores match expected ranges."""
        total_samples = 0
        compliant_samples = 0
        
        for _, row in self.df.iterrows():
            score = self.metric(row['text1'], row['text2'])
            expected_range = row['expected_range']
            
            if expected_range[0] <= score <= expected_range[1]:
                compliant_samples += 1
            total_samples += 1
        
        return {
            'compliance_rate': float(compliant_samples / total_samples),
            'total_samples': total_samples,
            'compliant_samples': compliant_samples
        }

    def _calculate_overall_stats(self) -> Dict[str, float]:
        """Calculate overall statistical measures."""
        all_scores = []
        expected_means = []
        
        for _, row in self.df.iterrows():
            score = self.metric(row['text1'], row['text2'])
            all_scores.append(score)
            expected_means.append(np.mean(row['expected_range']))
        
        return {
            'mean_score': float(np.mean(all_scores)),
            'std_score': float(np.std(all_scores)),
            'correlation_with_expected': float(np.corrcoef(
                all_scores, expected_means
            )[0,1])
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to JSON."""
        output_path = self.output_dir / "characterization_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")