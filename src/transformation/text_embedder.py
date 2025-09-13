"""
Text embedding generation using Sentence Transformers.
"""
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

from utils.logger import BaseLogger

class TextEmbedder:
    """Class for generating text embeddings using pre-trained models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", logger: Optional[BaseLogger] = None):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the pre-trained model to use
            logger: Logger instance for logging
        """
        self.model_name = model_name
        self.logger = logger or BaseLogger("TextEmbedder")
        self.model = None
        self.embedding_dim = 0
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pre-trained model."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension from the model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array with the text embedding
        """
        if not text or not isinstance(text, str):
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array with embeddings for all texts
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and isinstance(text, str)]
        if not valid_texts:
            return np.zeros((len(texts), self.embedding_dim))
        
        # Generate embeddings
        embeddings = self.model.encode(valid_texts, batch_size=batch_size, 
                                     convert_to_numpy=True)
        
        # Handle cases where some texts were invalid
        if len(valid_texts) != len(texts):
            result = np.zeros((len(texts), self.embedding_dim))
            valid_idx = 0
            for i, text in enumerate(texts):
                if text and isinstance(text, str):
                    result[i] = embeddings[valid_idx]
                    valid_idx += 1
            return result
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dim

class TextEmbeddingManager:
    """Manager class for handling text embedding operations."""
    
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        self.embeddings_cache = {}  # Simple cache for embeddings
    
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for text, with optional caching.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Text embedding
        """
        if not text:
            return np.zeros(self.embedder.get_embedding_dimension())
        
        if use_cache and text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = self.embedder.embed_text(text)
        
        if use_cache:
            self.embeddings_cache[text] = embedding
        
        return embedding
    
    def precompute_embeddings(self, texts: List[str], description: str = "") -> np.ndarray:
        """
        Precompute embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            description: Description for logging
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        self.embedder.logger.info(f"Precomputing embeddings for {len(texts)} {description}")
        return self.embedder.embed_batch(texts)
    
    def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        self.embeddings_cache.clear()

