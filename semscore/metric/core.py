"""Core implementation of SemScore metric."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Union, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class SemScoreMetric:
    """Semantic similarity metric using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_size: int = 10000,
        device: Optional[str] = None
    ):
        """Initialize metric with model and device placement."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Setup caching
        self.cache_size = cache_size
        self._setup_cache()

    def _setup_cache(self) -> None:
        """Configure LRU cache for embeddings."""
        self.get_cached_embedding = lru_cache(maxsize=self.cache_size)(
            self._get_embedding
        )

    def mean_pooling(self, model_output, attention_mask):
        """Calculate mean pooled embeddings with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        # Use simple sum and divide for mean pooling
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Generate normalized embedding for input text."""
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = self.model(**encoded)
            sentence_embedding = self.mean_pooling(output, encoded['attention_mask'])
            # Normalize to unit length
            return F.normalize(sentence_embedding, p=2, dim=1)

    def __call__(
        self, 
        text1: Union[str, Dict[str, str]], 
        text2: Union[str, Dict[str, str]], 
        trace: Optional[Dict] = None
    ) -> float:
        """Calculate semantic similarity between two texts."""
        # Handle DSPy-style inputs
        if isinstance(text1, dict) and isinstance(text2, dict):
            if 'response' not in text1 or 'response' not in text2:
                raise ValueError("Missing 'response' key in input dict")
            text1, text2 = text1['response'], text2['response']
        
        # Input validation
        if not isinstance(text1, str) or not isinstance(text2, str):
            raise ValueError("Inputs must be strings")
        if not text1.strip() or not text2.strip():
            raise ValueError("Inputs cannot be empty")
        
        # Get embeddings using cached function
        emb1 = self.get_cached_embedding(text1)
        emb2 = self.get_cached_embedding(text2)
        
        # Calculate cosine similarity directly
        similarity = float(F.cosine_similarity(emb1, emb2).item())
        
        # Populate trace if provided
        if trace is not None:
            trace['semantic_similarity'] = similarity
            trace['embeddings'] = {
                'prediction': emb1.cpu().numpy(),
                'reference': emb2.cpu().numpy()
            }
        
        return similarity

def semscore_metric(
    pred: Dict[str, Any], 
    example: Dict[str, Any], 
    trace: Optional[Dict] = None
) -> float:
    """DSPy-compatible metric function."""
    if not hasattr(semscore_metric, '_scorer'):
        semscore_metric._scorer = SemScoreMetric()
        
    # Input validation
    assert 'response' in pred and 'response' in example, "Missing response key"
    assert isinstance(pred['response'], str) and isinstance(example['response'], str), "Non-string response"
    assert len(pred['response'].strip()) > 0 and len(example['response'].strip()) > 0, "Empty response"
    
    # Get embeddings
    pred_embedding = semscore_metric._scorer.get_cached_embedding(pred['response'])
    ref_embedding = semscore_metric._scorer.get_cached_embedding(example['response'])
    
    # Calculate similarity directly
    similarity = float(F.cosine_similarity(pred_embedding, ref_embedding).item())
    
    # Populate trace if provided
    if trace is not None:
        trace['semantic_similarity'] = similarity
        trace['embeddings'] = {
            'prediction': pred_embedding.cpu().numpy(),
            'reference': ref_embedding.cpu().numpy()
        }
    
    return similarity