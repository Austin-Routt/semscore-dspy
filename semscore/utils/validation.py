"""
Validation utilities for SemScore metric.
Handles validation against OpenAssistant dataset format.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from scipy import stats

from ..visualization.plots import SemScorePlotter
from ..metric.core import SemScoreMetric, semscore_metric

logger = logging.getLogger(__name__)

class SemScoreValidator:
    """Validator for SemScore implementation."""
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "semscore_results",
        reference_file: Optional[Union[str, Path]] = None
    ):
        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up reference data path
        if reference_file is None:
            self.reference_file = Path(__file__).parent.parent / "data" / "reference" / "semscores_OA-100.json"
        else:
            self.reference_file = Path(reference_file)
            
        # Initialize visualization helper and timestamp
        self.plotter = SemScorePlotter(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized validator with output directory: {self.output_dir}")


    def debug_reference_file(self):
        """Debug reference file loading."""
        try:
            with open(self.reference_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded reference file")
                print(f"Number of models: {len(data)}")
                for model, cases in data.items():
                    print(f"\nModel: {model}")
                    print(f"Number of cases: {len(cases)}")
                    print("First case keys:", list(cases[0].keys()))
        except Exception as e:
            print(f"Error loading reference file: {e}")

    def debug_case(self, case: Dict):
        """Debug processing of a single case."""
        try:
            pred_dict = {'response': case['answer_pred']}
            ref_dict = {'response': case['answer_ref']}
            score = semscore_metric(pred_dict, ref_dict)
            print(f"Successfully processed case")
            print(f"Calculated score: {score}")
            print(f"Expected score: {case['cosine_sim']}")
        except Exception as e:
            print(f"Error processing case: {e}")


    def validate_against_references(self) -> Dict[str, Any]:
        """Validate implementation against reference scores."""
        logger.info("Starting validation against reference scores...")
        
        # Load and validate reference data
        try:
            with open(self.reference_file, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
                logger.debug(f"Loaded reference data: {len(reference_data)} models")
        except Exception as e:
            logger.error(f"Error loading reference file: {e}")
            raise
        
        results = {}
        for model_name, test_cases in reference_data.items():
            logger.info(f"Processing {model_name} with {len(test_cases)} test cases")
            
            calculated_scores = []
            expected_scores = []
            examples = []
            
            for idx, case in enumerate(test_cases):
                try:
                    # Validate case structure
                    for key in ['answer_pred', 'answer_ref', 'cosine_sim']:
                        if key not in case:
                            logger.error(f"Missing key '{key}' in case {idx}")
                            continue
                    
                    # Calculate scores
                    pred_dict = {'response': case['answer_pred']}
                    ref_dict = {'response': case['answer_ref']}
                    
                    calc_score = semscore_metric(pred_dict, ref_dict)
                    expected_score = float(case['cosine_sim'])
                    
                    calculated_scores.append(calc_score)
                    expected_scores.append(expected_score)
                    
                    logger.debug(f"Case {idx}: calc={calc_score:.4f}, expected={expected_score:.4f}")
                    
                    # Store example
                    if len(examples) < 3:
                        examples.append({
                            'prompt': case.get('prompt', 'No prompt available'),
                            'pred': case['answer_pred'][:200] + '...',
                            'ref': case['answer_ref'][:200] + '...',
                            'calculated': calc_score,
                            'expected': expected_score,
                            'diff': abs(calc_score - expected_score)
                        })
                
                except Exception as e:
                    logger.error(f"Error processing case {idx}: {e}")
                    continue
            
            # Process results if we have valid scores
            if calculated_scores and expected_scores:
                logger.info(f"Processed {len(calculated_scores)} valid cases for {model_name}")
                results[model_name] = {
                    'calculated_stats': self.analyze_distribution(
                        calculated_scores, 
                        f"{model_name.replace('/', '_')}_calculated"
                    ),
                    'expected_stats': self.analyze_distribution(
                        expected_scores, 
                        f"{model_name.replace('/', '_')}_expected"
                    ),
                    'correlation': float(stats.pearsonr(calculated_scores, expected_scores)[0]),
                    'mean_difference': float(np.mean(np.abs(
                        np.array(calculated_scores) - np.array(expected_scores)
                    ))),
                    'examples': examples
                }
            else:
                logger.error(f"No valid results for {model_name}")
        
        if not results:
            raise ValueError("No valid results generated from reference data")
            
        return results

    def analyze_distribution(self, scores: List[float], name: str) -> Dict[str, float]:
        """Analyze statistical properties of score distribution."""
        scores_array = np.array(scores)
        
        stats_dict = {
            'mean': float(np.mean(scores_array)),
            'median': float(np.median(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array))
        }
        
        # Only calculate higher moments if we have enough samples
        if len(scores_array) > 2:
            stats_dict.update({
                'skew': float(stats.skew(scores_array)),
                'kurtosis': float(stats.kurtosis(scores_array))
            })
        
        # Generate distribution plot
        self.plotter.plot_score_distribution(scores_array, name, self.timestamp)
        
        return stats_dict

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        reference_results = self.validate_against_references()
        
        # Generate report
        report = f"""
# SemScore Validation Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## OpenAssistant Dataset Results
"""
        
        for model_name, results in reference_results.items():
            report += f"\n### {model_name}\n\n"
            report += f"""
Performance Metrics:
- Pearson Correlation: {results['correlation']:.4f}
- Mean Absolute Difference: {results['mean_difference']:.4f}
- Mean Calculated Score: {results['calculated_stats']['mean']:.4f}
- Mean Reference Score: {results['expected_stats']['mean']:.4f}

Example Cases:
| Prompt | Prediction | Reference | Calculated | Expected | Diff |
|--------|------------|-----------|------------|-----------|------|
"""
            for ex in results['examples']:
                # Truncate text for readability
                prompt = ex['prompt'][:50].replace('\n', ' ') + '...'
                pred = ex['pred'][:50].replace('\n', ' ') + '...'
                ref = ex['ref'][:50].replace('\n', ' ') + '...'
                
                report += f"| {prompt} | {pred} | {ref} | {ex['calculated']:.4f} | {ex['expected']:.4f} | {ex['diff']:.4f} |\n"
            
        # Save report
        report_path = self.output_dir / f"validation_report_{self.timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report

def run_validation(
    output_dir: Optional[Union[str, Path]] = None,
    reference_file: Optional[Union[str, Path]] = None
) -> str:
    """Run complete validation suite."""
    validator = SemScoreValidator(output_dir, reference_file)
    report = validator.generate_report()
    logger.info(f"Validation complete. Report saved to {validator.output_dir}")
    return report