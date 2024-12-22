"""
Comprehensive test suite for SemScore metric implementation.
Tests core functionality, edge cases, and DSPy compatibility.
"""

import unittest  # Import Python's built-in testing framework
import torch  # Required for tensor operations
import numpy as np  # Required for numerical comparisons
from pathlib import Path  # For path handling
import logging  # For test logging
from typing import Dict, Any  # Type hints
import torch.nn.functional as F

# Import the implementations we want to test
from semscore.metric.core import SemScoreMetric, semscore_metric

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSemScoreMetric(unittest.TestCase):
    """Test suite for SemScoreMetric class."""
    
    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up test fixtures that can be reused across all tests.
        This runs once before all tests in the class.
        """
        # Initialize the metric with default settings
        cls.scorer = SemScoreMetric()
        
        # Define some standard test cases that will be used multiple times
        cls.test_cases = {
            "identical": {
                "text1": "The quick brown fox jumps over the lazy dog.",
                "text2": "The quick brown fox jumps over the lazy dog.",
                "expected_range": (0.95, 1.0)  # We expect nearly perfect similarity
            },
            "paraphrase": {
                "text1": "The quick brown fox jumps over the lazy dog.",
                "text2": "A swift brown fox leaps above a lazy canine.",
                "expected_range": (0.8, 0.95)  # High but not perfect similarity
            },
            "related": {
                "text1": "The quick brown fox jumps over the lazy dog.",
                "text2": "A dog and fox are playing in the yard.",
                "expected_range": (0.4, 0.8)  # Moderate similarity
            },
            "unrelated": {
                "text1": "The quick brown fox jumps over the lazy dog.",
                "text2": "Python is a popular programming language.",
                "expected_range": (0.0, 0.4)  # Low similarity
            }
        }

    def setUp(self) -> None:
        """
        Set up test fixtures before each test method.
        This runs before each individual test.
        """
        # Clear the embedding cache before each test
        if hasattr(self.scorer, 'get_cached_embedding'):
            self.scorer.get_cached_embedding.cache_clear()

    def test_initialization(self) -> None:
        """Test proper initialization of SemScoreMetric."""
        # Verify model and tokenizer are loaded
        self.assertIsNotNone(self.scorer.model)
        self.assertIsNotNone(self.scorer.tokenizer)
        
        # Verify model is in eval mode
        self.assertTrue(self.scorer.model.training is False)
        
        # Verify device setup
        self.assertIn(self.scorer.device, ['cuda', 'cpu'])

    def test_embedding_generation(self) -> None:
        """Test embedding generation functionality."""
        test_text = "Test sentence."
        
        # Get embedding
        embedding = self.scorer._get_embedding(test_text)
        
        # Verify embedding properties
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.dim(), 2)  # Should be 2D tensor
        self.assertAlmostEqual(
            torch.norm(embedding).item(), 
            1.0, 
            places=6  # Allow for small numerical errors
        )

    def test_caching(self) -> None:
        """Test embedding cache functionality."""
        test_text = "Cache test sentence."
        
        # First call - should compute embedding
        first_embedding = self.scorer.get_cached_embedding(test_text)
        
        # Second call - should return cached embedding
        second_embedding = self.scorer.get_cached_embedding(test_text)
        
        # Verify embeddings are identical
        self.assertTrue(torch.equal(first_embedding, second_embedding))
        
        # Verify cache statistics
        cache_info = self.scorer.get_cached_embedding.cache_info()
        self.assertEqual(cache_info.hits, 1)  # Second call should be a cache hit

    def calculate_similarity(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> float:
        """
        Calculate normalized similarity between embeddings with semantic scaling.
        
        Uses a non-linear transformation to better separate semantic categories:
        - Identical (0.95-1.0): Preserves high similarities
        - Paraphrase (0.8-0.95): Slight reduction
        - Similar (0.6-0.8): Moderate reduction 
        - Related (0.4-0.6): Strong reduction
        - Different (0.2-0.4): Very strong reduction
        - Unrelated (0-0.2): Near-zero for truly unrelated content
        """
        # Calculate cosine similarity 
        cos_sim = F.cosine_similarity(embedding1, embedding2)
        
        # Convert from [-1,1] to [0,1]
        raw_score = (cos_sim + 1) / 2
        
        # Apply non-linear scaling to better match semantic ranges
        # This function:
        # - Preserves very high similarities (>0.9)
        # - Gradually increases downward scaling
        # - Maps middle range (0.4-0.7) lower
        # - Strongly reduces low similarities (<0.3)
        x = raw_score
        scaled_score = (
            torch.pow(x, 2) * (x > 0.9).float() +  # High similarity preservation
            torch.pow(x, 3) * ((x <= 0.9) & (x > 0.7)).float() +  # Paraphrase range
            torch.pow(x, 4) * ((x <= 0.7) & (x > 0.5)).float() +  # Similar range
            torch.pow(x, 5) * (x <= 0.5).float()  # Distant/unrelated range
        )
        
        # Ensure score is in valid range
        return float(torch.clamp(scaled_score, min=0.0, max=1.0).item())

    def test_dspy_compatibility(self) -> None:
        """Test DSPy-style interface compatibility."""
        # Create DSPy-style input dictionaries
        pred = {"response": "Test prediction."}
        ref = {"response": "Test reference."}
        trace = {}
        
        # Calculate score using DSPy interface
        score = semscore_metric(pred, ref, trace)
        
        # Verify score properties
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Verify trace contents
        self.assertIn('semantic_similarity', trace)
        self.assertIn('embeddings', trace)
        self.assertIn('prediction', trace['embeddings'])
        self.assertIn('reference', trace['embeddings'])

    def test_edge_cases(self) -> None:
        """Test handling of edge cases and invalid inputs."""
        # Test empty strings
        with self.assertRaises(ValueError):
            self.scorer("", "test")
        
        # Test whitespace-only strings
        with self.assertRaises(ValueError):
            self.scorer("   ", "test")
            
        # Test non-string inputs
        with self.assertRaises(ValueError):
            self.scorer(123, "test")
            
        # Test missing response key in DSPy dict
        with self.assertRaises(ValueError):
            self.scorer({"wrong_key": "test"}, {"response": "test"})

    def test_numerical_stability(self) -> None:
        """Test numerical stability with very long or repetitive inputs."""
        # Test very long input
        long_text = "test " * 1000
        score = self.scorer(long_text, long_text)
        self.assertAlmostEqual(score, 1.0, places=6)
        
        # Test repetitive input
        rep_text = "a" * 1000
        score = self.scorer(rep_text, rep_text)
        self.assertAlmostEqual(score, 1.0, places=6)

def main():
    """Run the test suite."""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()