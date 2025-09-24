import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from src.utils.logging_utils import PerformanceLogger
from src.utils.file_utils import ensure_dir
from src.exceptions.core_exceptions import SimilarityMatchingError

logger = logging.getLogger(__name__)

class SimilarityMatch:
    """data class for similarity match results"""
    def __init__(self, product_id: str, similarity: float, embedding_index: int,
                 product_name: str = None, metadata: Dict = None):
        self.product_id = product_id
        self.similarity = similarity
        self.embedding_index = embedding_index
        self.product_name = product_name
        self.metadata = metadata
        
    def to_dict(self) -> Dict:
        """Convert match to dictionary"""
        return{
            "product_id": self.product_id,
            "similarity": self.similarity,
            "embedding_index": self.embedding_index,
            "product_name": self.product_name,
            "metadata": self.metadata
        }
        
        
class SimilarityMatcher:
    """FAISS-based similarity matching with multiple index types"""
    def __init__(self, embedding_dim: int, index_type: str = "IndexFlatIP"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.product_ids = []
        self.product_metadata = {}
        
        logger.info(f"Initializing similarity matcher: {index_type} for {embedding_dim}D embeddings")
        self._create_index()
        
    def _create_index(self) -> None:
        """Create FAISS index"""
        try:
            if self.index_type == "IndexFlatIP":
                # Inner product (for cosine similarity with normalized vectors)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                
            elif self.index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                
            elif self.index_type == "IndexIVFFlat":
                # Inverted file index (for large catalogs)
                nlist = 100  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
            elif self.index_type == "IndexHNSWFlat":
                # Hierarchical NSW for fast approximate search
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info(f"Created FAISS index: {self.index_type}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise SimilarityMatchingError(f"Index creation failed: {e}")
        
    def add_embedding(self, embedding: np.ndarray, product_id: str, 
                     metadata: Dict = None) -> None:
        """Add embedding to index"""
        try:
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: {embedding.shape[0]} vs {self.embedding_dim}")
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
            
            # Store metadata
            self.product_ids.append(product_id)
            self.product_metadata[product_id] = metadata or {}
            
            logger.debug(f"Added embedding for product: {product_id}")
            
        except Exception as e:
            logger.error(f"Failed to add embedding for {product_id}: {e}")
            raise SimilarityMatchingError(f"Add embedding failed: {e}")
        
    def search(self, query_embedding: np.ndarray, k: int = 1, 
              similarity_threshold: float = 0.0) -> List[SimilarityMatch]:
        try:
            
            if self.index.ntotal == 0:
                return []
            if query_embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Query embedding dimension mismatch: {query_embedding.shape[0]} vs {self.embedding_dim}")

            # Perform search
            similarities, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32), k
            )
            
            matches = []
            for sim, idx in zip(similarities[0], indices[0]):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.product_ids):
                    continue
                
                # Apply similarity threshold
                if sim < similarity_threshold:
                    continue
                
                product_id = self.product_ids[idx]
                match = SimilarityMatch(
                    product_id=product_id,
                    similarity=float(sim),
                    embedding_index=int(idx),
                    product_name=self.product_metadata[product_id].get('name', product_id),
                    metadata=self.product_metadata[product_id]
                )
                
                matches.append(match)
            
            logger.debug(f"Found {len(matches)} matches above threshold {similarity_threshold}")
            return matches
        
        
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise SimilarityMatchingError(
                f"Search failed: {e}",
                embedding_shape=query_embedding.shape,
                catalog_size=self.index.ntotal
            )
            
    def train_index(self) -> None:
        """Train index (required for some index types like IVF)"""
        try:
            if hasattr(self.index, 'train') and not self.index.is_trained:
                logger.info("Training FAISS index...")
                # For IVF indices, we need training data
                if self.index.ntotal > 0:
                    training_data = self.index.reconstruct_n(0, self.index.ntotal)
                    self.index.train(training_data)
                    logger.info("✅ Index training completed")
                else:
                    logger.warning("No data available for training")
            
        except Exception as e:
            logger.error(f"Index training failed: {e}")
            raise SimilarityMatchingError(f"Training failed: {e}")
        
    def save_index(self, filepath: Path) -> None:
        """Save FAISS index and metadata"""
        try:
            ensure_dir(Path(filepath).parent)
            
            # Save FAISS index
            faiss.write_index(self.index, str(filepath))
            
            # Save metadata
            metadata_file = Path(filepath).with_suffix('.metadata.pkl')
            metadata = {
                'product_ids': self.product_ids,
                'product_metadata': self.product_metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }
            
            import pickle
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise SimilarityMatchingError(f"Save failed: {e}")
        
        
    def load_index(self, filepath: Path) -> None:
        """Load FAISS index and metadata"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(filepath))
            
            # Load metadata
            metadata_file = Path(filepath).with_suffix('.metadata.pkl')
            
            import pickle
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.product_ids = metadata['product_ids']
            self.product_metadata = metadata['product_metadata']
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            
            logger.info(f"Loaded index from {filepath}: {len(self.product_ids)} products")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise SimilarityMatchingError(f"Load failed: {e}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get index statistics"""
        return {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'total_embeddings': self.index.ntotal if self.index else 0,
            'total_products': len(self.product_ids),
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
    
    def clear(self) -> None:
        """Clear all data from index"""
        self.product_ids.clear()
        self.product_metadata.clear()
        self._create_index()  # Recreate empty index
        logger.info("Index cleared")