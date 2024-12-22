"""
Tests for SemScore metric and validation utilities.
Includes test cases for both basic functionality and reference data validation.
"""

import unittest
import tempfile
from pathlib import Path
import json
import shutil
import numpy as np
from typing import Dict, Any
import logging

from semscore.utils.validation import SemScoreValidator
from semscore.metric.core import semscore_metric

logger = logging.getLogger(__name__)

class TestSemScoreValidation(unittest.TestCase):
    """
    Comprehensive test suite for SemScore validation.
    Tests both basic functionality and reference data alignment.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test fixtures that will be used across all tests.
        Creates temporary directories and sample data.
        """
        # Create temporary test directory
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample reference data matching OA format
        cls.test_reference_data = {
            "test_model": [
                {
                    "prompt": "<|user|>\nTest question?</s>\n<|assistant|>\n",
                    "answer_pred": "This is a long detailed answer that would come from the model being tested.",
                    "answer_ref": "This is the reference answer that we would expect from a good model.",
                    "cosine_sim": 0.85
                },
                {
                    "prompt": "<|user|>\nAnother question?</s>\n<|assistant|>\n",
                    "answer_pred": "Here is another detailed answer with specific information and context.",
                    "answer_ref": "Here is what we would expect as a good response with proper details.",
                    "cosine_sim": 0.90
                }
            ]
        }
        
        # Save test reference data
        cls.test_reference_file = cls.test_dir / "test_references.json"
        with open(cls.test_reference_file, 'w', encoding='utf-8') as f:
            json.dump(cls.test_reference_data, f)
            
        # Load real reference data for validation tests
        cls.reference_file = Path("semscore/data/reference/semscores_OA-100.json")
        if not cls.reference_file.exists():
            raise FileNotFoundError("Reference data file not found")
            
        with open(cls.reference_file, 'r', encoding='utf-8') as f:
            cls.reference_data = json.load(f)
            
        # Initialize validators
        cls.test_validator = SemScoreValidator(
            output_dir=cls.test_dir,
            reference_file=cls.test_reference_file
        )
        cls.reference_validator = SemScoreValidator(
            output_dir=cls.test_dir,
            reference_file=cls.reference_file
        )
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir)
        
    def test_basic_initialization(self):
        """Test proper initialization of validator."""
        self.assertTrue(self.test_validator.output_dir.exists())
        self.assertTrue(self.test_validator.reference_file.exists())
        
    def test_reference_data_structure(self):
        """Test that reference data has correct structure."""
        # Test structure of test data
        self.assertTrue(isinstance(self.test_reference_data, dict))
        self.assertTrue(len(self.test_reference_data) > 0)
        
        # Test structure of real reference data
        self.assertTrue(isinstance(self.reference_data, dict))
        self.assertTrue(len(self.reference_data) > 0)
        
        # Validate data format
        for model, cases in self.reference_data.items():
            self.assertTrue(isinstance(cases, list))
            self.assertTrue(len(cases) > 0)
            
            # Check first case structure
            case = cases[0]
            required_keys = {'prompt', 'answer_pred', 'answer_ref', 'cosine_sim'}
            self.assertTrue(all(k in case for k in required_keys))
            
    def test_score_calculation(self):
        """Test that score calculation matches reference values."""
        for model, cases in self.reference_data.items():
            for case in cases[:5]:  # Test first 5 cases per model
                # Calculate score using our implementation
                pred = {'response': case['answer_pred']}
                ref = {'response': case['answer_ref']}
                calculated_score = semscore_metric(pred, ref)
                
                # Compare with reference score
                expected_score = case['cosine_sim']
                
                # Verify score is within acceptable range
                # We use a tolerance because of potential floating point differences
                self.assertAlmostEqual(
                    calculated_score,
                    expected_score,
                    delta=0.1,  # Allow 0.1 difference
                    msg=f"Score mismatch for model {model}"
                )
                
    def test_statistical_metrics(self):
        """Test statistical metrics calculation."""
        # Run validation on reference data
        results = self.reference_validator.validate_against_references()
        
        for model, model_results in results.items():
            # Check basic statistics presence
            self.assertIn('calculated_stats', model_results)
            self.assertIn('expected_stats', model_results)
            
            # Verify statistical measures
            stats = model_results['calculated_stats']
            self.assertIn('mean', stats)
            self.assertIn('median', stats)
            self.assertIn('std', stats)
            
            # Verify correlation
            self.assertIn('correlation', model_results)
            correlation = model_results['correlation']
            self.assertGreaterEqual(correlation, -1.0)
            self.assertLessEqual(correlation, 1.0)
            
            # Verify mean difference
            self.assertIn('mean_difference', model_results)
            mean_diff = model_results['mean_difference']
            self.assertGreaterEqual(mean_diff, 0.0)
            
    def test_example_storage(self):
        """Test that examples are properly stored and formatted."""
        results = self.reference_validator.validate_against_references()
        
        for model, model_results in results.items():
            # Verify examples are stored
            self.assertIn('examples', model_results)
            examples = model_results['examples']
            
            # Check number of examples
            self.assertLessEqual(len(examples), 3)  # Should store max 3 examples
            
            # Check example format
            for example in examples:
                required_keys = {'prompt', 'pred', 'ref', 'calculated', 'expected', 'diff'}
                self.assertTrue(all(k in example for k in required_keys))
                
    def test_score_distribution(self):
        """Test distribution of calculated scores vs reference scores."""
        results = self.reference_validator.validate_against_references()
        
        for model, model_results in results.items():
            calc_stats = model_results['calculated_stats']
            exp_stats = model_results['expected_stats']
            
            # Verify mean scores are within reasonable range
            self.assertGreater(calc_stats['mean'], 0.0)
            self.assertLess(calc_stats['mean'], 1.0)
            self.assertGreater(exp_stats['mean'], 0.0)
            self.assertLess(exp_stats['mean'], 1.0)
            
            # Verify standard deviations are reasonable
            self.assertGreater(calc_stats['std'], 0.0)
            self.assertLess(calc_stats['std'], 1.0)
            
    def test_report_generation(self):
        """Test report generation with reference data."""
        report = self.reference_validator.generate_report()
        
        # Verify report structure
        self.assertIsInstance(report, str)
        self.assertIn("SemScore Validation Report", report)
        
        # Verify model results are included
        for model in self.reference_data.keys():
            self.assertIn(model, report)
            
        # Verify metrics are included
        self.assertIn("Pearson Correlation:", report)
        self.assertIn("Mean Absolute Difference:", report)
        self.assertIn("Example Cases:", report)

def main():
    """Run the test suite."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()