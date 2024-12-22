"""Test suite for SemScore characterization module."""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from jsonschema import ValidationError
import logging
from typing import Dict, List, Any

from semscore.characterization.core import CharacterizationSuite
from semscore.metric.core import SemScoreMetric

logger = logging.getLogger(__name__)

class TestCharacterizationSuite(unittest.TestCase):
    """Test cases for CharacterizationSuite class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across all tests."""
        # Create temporary test directory
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create sample test data
        cls.test_data = [
            {
                "category": "Identical",
                "domain": "Technical",
                "text1": "The sky is blue.",
                "text2": "The sky is blue.",
                "expected_range": [0.95, 1.0],
                "source": "Test Data",
                "subdomain": "Basic Facts"
            },
            {
                "category": "Paraphrase",
                "domain": "Technical", 
                "text1": "The quick brown fox jumps over the lazy dog.",
                "text2": "A swift brown fox leaps above a lazy canine.",
                "expected_range": [0.8, 0.95],
                "source": "Test Data",
                "subdomain": "Common Phrases"
            },
            {
                "category": "Unrelated",
                "domain": "Technical",
                "text1": "Python is a programming language.",
                "text2": "Apples are delicious.",
                "expected_range": [0.0, 0.2],
                "source": "Test Data",
                "subdomain": "Mixed Topics"
            }
        ]
        
        # Create test schema
        cls.test_schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "Identical",
                        "Paraphrase",
                        "Similar Content",
                        "Related Topic",
                        "Different Domain",
                        "Unrelated",
                        "Contradiction"
                    ]
                },
                "domain": {
                    "type": "string",
                    "enum": [
                        "Technical",
                        "Creative",
                        "Conversational",
                        "Academic",
                        "News",
                        "Code"
                    ]
                },
                "text1": {"type": "string", "minLength": 1},
                "text2": {"type": "string", "minLength": 1},
                "expected_range": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["category", "domain", "text1", "text2", "expected_range"]
        }
        
        # Save test files
        cls.schema_path = cls.test_dir / "test_schema.json"
        cls.dataset_path = cls.test_dir / "test_dataset.json"
        
        with open(cls.schema_path, 'w', encoding='utf-8') as f:
            json.dump(cls.test_schema, f)
            
        with open(cls.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(cls.test_data, f)
            
        # Create output directory for test results
        cls.output_dir = cls.test_dir / "test_results"
        cls.output_dir.mkdir(exist_ok=True)

        cls.shared_metric = SemScoreMetric()
        
        # Initialize main test suite with shared metric
        cls.suite = CharacterizationSuite(
            schema_path=str(cls.schema_path),
            dataset_path=str(cls.dataset_path),
            output_dir=str(cls.output_dir),
            metric=cls.shared_metric
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Set up test fixtures before each test."""
        # Clear any cached data
        if hasattr(self.suite.metric, 'get_cached_embedding'):
            self.suite.metric.get_cached_embedding.cache_clear()

    def test_initialization(self):
        """Test proper initialization of CharacterizationSuite."""
        self.assertIsNotNone(self.suite.metric)
        self.assertIsInstance(self.suite.df, pd.DataFrame)
        self.assertEqual(len(self.suite.df), len(self.test_data))
        self.assertTrue(self.output_dir.exists())

    def test_invalid_schema(self):
        """Test handling of invalid schema."""
        # Create mock metric that doesn't load a real model
        class MockMetric:
            def __call__(self, text1, text2):
                return 0.5
        
        # Create an invalid schema that will trigger validation error but is valid JSON Schema
        invalid_schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": []  # Empty enum makes this invalid for our use case
                },
                "domain": {
                    "type": "number"  # Wrong type for our domain validation
                }
            }
        }

        invalid_schema_path = self.test_dir / "invalid_schema.json"
        with open(invalid_schema_path, 'w') as f:
            json.dump(invalid_schema, f)
            
        # This should now raise ValidationError when validating dataset against this schema
        with self.assertRaises(ValidationError):
            CharacterizationSuite(
                schema_path=str(invalid_schema_path),
                dataset_path=str(self.dataset_path),
                metric=MockMetric()
            )

    def test_invalid_dataset(self):
        """Test handling of invalid dataset."""
        invalid_dataset_path = self.test_dir / "invalid_dataset.json"
        invalid_data = [{"invalid": "data"}]
        
        with open(invalid_dataset_path, 'w') as f:
            json.dump(invalid_data, f)
            
        with self.assertRaises(ValidationError):
            CharacterizationSuite(
                schema_path=str(self.schema_path),
                dataset_path=str(invalid_dataset_path)
            )

    def test_category_analysis(self):
        """Test category analysis functionality."""
        results = self.suite._analyze_by_category()
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        
        # Check each category's results
        for category in self.test_data:
            cat_name = category['category']
            if cat_name in results:
                result = results[cat_name]
                
                # Verify required statistics
                self.assertIn('mean_score', result)
                self.assertIn('std_score', result)
                self.assertIn('median_score', result)
                self.assertIn('range_compliance', result)
                self.assertIn('sample_count', result)
                
                # Verify value ranges
                self.assertGreaterEqual(result['mean_score'], 0.0)
                self.assertLessEqual(result['mean_score'], 1.0)
                self.assertGreaterEqual(result['sample_count'], 1)

    def test_domain_analysis(self):
        """Test domain analysis functionality."""
        results = self.suite._analyze_by_domain()
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        
        # Check each domain's results
        for domain in set(item['domain'] for item in self.test_data):
            self.assertIn(domain, results)
            result = results[domain]
            
            # Verify required statistics
            self.assertIn('mean_score', result)
            self.assertIn('std_score', result)
            self.assertIn('score_by_category', result)
            self.assertIn('sample_count', result)
            
            # Verify value ranges
            self.assertGreaterEqual(result['mean_score'], 0.0)
            self.assertLessEqual(result['mean_score'], 1.0)
            self.assertGreaterEqual(result['sample_count'], 1)

    def test_range_compliance(self):
        """Test range compliance analysis."""
        results = self.suite._analyze_range_compliance()
        
        # Verify results structure
        self.assertIn('compliance_rate', results)
        self.assertIn('total_samples', results)
        self.assertIn('compliant_samples', results)
        
        # Verify value ranges
        self.assertGreaterEqual(results['compliance_rate'], 0.0)
        self.assertLessEqual(results['compliance_rate'], 1.0)
        self.assertEqual(results['total_samples'], len(self.test_data))
        self.assertGreaterEqual(results['compliant_samples'], 0)
        self.assertLessEqual(results['compliant_samples'], results['total_samples'])

    def test_overall_stats(self):
        """Test overall statistics calculation."""
        results = self.suite._calculate_overall_stats()
        
        # Verify results structure
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        self.assertIn('correlation_with_expected', results)
        
        # Verify value ranges
        self.assertGreaterEqual(results['mean_score'], 0.0)
        self.assertLessEqual(results['mean_score'], 1.0)
        self.assertGreaterEqual(results['std_score'], 0.0)
        self.assertGreaterEqual(results['correlation_with_expected'], -1.0)
        self.assertLessEqual(results['correlation_with_expected'], 1.0)

    def test_full_characterization(self):
        """Test complete characterization workflow."""
        results = self.suite.run_characterization()
        
        # Verify results structure
        self.assertIn('category_analysis', results)
        self.assertIn('domain_analysis', results)
        self.assertIn('range_compliance', results)
        self.assertIn('overall_stats', results)
        
        # Verify output file creation
        results_file = self.output_dir / "characterization_results.json"
        self.assertTrue(results_file.exists())
        
        # Verify file contents
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        self.assertEqual(saved_results, results)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test empty strings (should be caught by schema validation)
        invalid_data = self.test_data.copy()
        invalid_data.append({
            "category": "Identical",
            "domain": "Technical",
            "text1": "",  # Empty string
            "text2": "Test",
            "expected_range": [0.95, 1.0]
        })
        
        invalid_path = self.test_dir / "invalid_data.json"
        with open(invalid_path, 'w') as f:
            json.dump(invalid_data, f)
            
        with self.assertRaises(ValidationError):
            CharacterizationSuite(
                schema_path=str(self.schema_path),
                dataset_path=str(invalid_path)
            )

    def test_performance(self):
        """Test performance with larger dataset."""
        # Create larger test dataset
        large_data = self.test_data * 10  # 30 samples
        
        large_path = self.test_dir / "large_dataset.json"
        with open(large_path, 'w') as f:
            json.dump(large_data, f)
            
        # Use existing shared metric instance
        suite = CharacterizationSuite(
            schema_path=str(self.schema_path),
            dataset_path=str(large_path),
            metric=self.shared_metric
        )
        
        # Verify performance is reasonable
        import time
        start_time = time.time()
        suite.run_characterization()
        end_time = time.time()
        
        # Should process at least 1 sample per second
        processing_time = end_time - start_time
        self.assertLess(processing_time / len(large_data), 1.0)

def main():
    """Run the test suite."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()