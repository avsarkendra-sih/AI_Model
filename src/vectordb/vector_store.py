import numpy as np
import json
import pickle
import os
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

from config.config import Config
from src.utils.logger import get_logger

class VectorStore:
    """Vector database interface for storing and retrieving embeddings."""
    
    def __init__(self, store_type="chroma"):
        """Initialize vector store."""
        self.logger = get_logger()
        self.config = Config()
        self.store_type = store_type
        
        if store_type == "chroma":
            self._init_chroma()
        elif store_type == "faiss":
            self._init_faiss()
        else:
            self._init_simple_store()
    
    def _init_chroma(self):
        """Initialize ChromaDB."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {str(e)}")
            self._init_simple_store()
    
    def _init_faiss(self):
        """Initialize FAISS (placeholder - would need full implementation)."""
        try:
            # For now, fall back to simple store
            # Full FAISS implementation would go here
            self.logger.info("FAISS not fully implemented, using simple store")
            self._init_simple_store()
            
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {str(e)}")
            self._init_simple_store()
    
    def _init_simple_store(self):
        """Initialize simple in-memory vector store."""
        self.store_type = "simple"
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Try to load existing data
        self._load_simple_store()
        
        self.logger.info("Simple vector store initialized")
    
    def _load_simple_store(self):
        """Load simple store from disk."""
        try:
            store_path = os.path.join("data", "simple_vector_store.pkl")
            if os.path.exists(store_path):
                with open(store_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])
                
                self.logger.info(f"Loaded {len(self.documents)} documents from simple store")
        except Exception as e:
            self.logger.error(f"Error loading simple store: {str(e)}")
    
    def _save_simple_store(self):
        """Save simple store to disk."""
        try:
            os.makedirs("data", exist_ok=True)
            store_path = os.path.join("data", "simple_vector_store.pkl")
            
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            
            with open(store_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            self.logger.error(f"Error saving simple store: {str(e)}")
    
    def add_document(self, doc_id: str, content: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add document to vector store."""
        try:
            if metadata is None:
                metadata = {}
            
            metadata['doc_id'] = doc_id
            
            if self.store_type == "chroma":
                self._add_to_chroma(doc_id, content, embedding, metadata)
            else:
                self._add_to_simple_store(doc_id, content, embedding, metadata)
                
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            raise
    
    def _add_to_chroma(self, doc_id: str, content: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add document to ChromaDB."""
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )
    
    def _add_to_simple_store(self, doc_id: str, content: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add document to simple store."""
        self.documents.append(content)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        # Save to disk
        self._save_simple_store()
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            if self.store_type == "chroma":
                return self._search_chroma(query_embedding, top_k)
            else:
                return self._search_simple_store(query_embedding, top_k)
                
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search ChromaDB."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            documents = []
            for i in range(len(results['ids'][0])):
                documents.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'doc_id': results['ids'][0][i]
                })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error searching ChromaDB: {str(e)}")
            return []
    
    def _search_simple_store(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search simple store."""
        if not self.embeddings:
            return []
        
        # Calculate similarities
        embeddings_array = np.array(self.embeddings)
        similarities = cosine_similarity([query_embedding], embeddings_array)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx]),
                'doc_id': self.metadata[idx].get('doc_id', f'doc_{idx}')
            })
        
        return results
    
    def get_document_count(self) -> int:
        """Get number of documents in store."""
        if self.store_type == "chroma":
            return self.collection.count()
        else:
            return len(self.documents)
    
    def delete_document(self, doc_id: str):
        """Delete document from store."""
        try:
            if self.store_type == "chroma":
                self.collection.delete(ids=[doc_id])
            else:
                # Find and remove from simple store
                indices_to_remove = []
                for i, metadata in enumerate(self.metadata):
                    if metadata.get('doc_id') == doc_id:
                        indices_to_remove.append(i)
                
                # Remove in reverse order to maintain indices
                for idx in reversed(indices_to_remove):
                    del self.documents[idx]
                    del self.embeddings[idx]
                    del self.metadata[idx]
                
                self._save_simple_store()
                
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        return {
            "store_type": self.store_type,
            "document_count": self.get_document_count(),
            "persist_directory": getattr(self.config, 'CHROMA_PERSIST_DIRECTORY', None)
        }
