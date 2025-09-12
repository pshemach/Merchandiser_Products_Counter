import logging
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import deque
from transformers import AutoImageProcessor, AutoModel
from src.utils.file_utils import save_pickle, load_pickle
from src.utils.image_utils import validate_image
from src.exceptions.core_exceptions import EmbeddingExtractionError

logger = logging.getLogger(__name__)

class EmbeddingStats:
    """Statistics tracking for embedding normalization"""
    def __init__(self):
        self.mean = None
        self.std = None
        self.count = None
        self.running_mean = None
        self.running_var = None
    
    def update(self, embedding: np.ndarray) -> None:
        """Update statistics with new embedding"""
        if self.mean is None:
            self.mean = embedding.copy()
            self.std = np.zeros_like(embedding)
            self.count = 1
            self.running_mean = embedding.copy()
            self.running_var = np.zeros_like(embedding)
            
        else:
            # Update using Welford's algorithm
            self.count += 1
            delta = embedding - self.running_mean
            self.running_mean += delta / self.count
            delta2 = embedding - self.running_mean
            self.running_var += delta * delta2
            
            # Update final statistics
            self.mean = self.running_mean.copy()
            if self.count > 1:
                self.std = np.sqrt(self.running_var / (self.count - 1))
                
    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding using stored statistics"""
        if self.mean is None or self.std is None:
            # Fallback to L2 normalization
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return (embedding - self.mean) / (self.std + 1e-8)
    
    def save(self, filepath: Path) -> None:
        """Save statistics to file"""
        stat_data = {
            'mean': self.mean,
            'std': self.std,
            'count':self.count
        }
        save_pickle(stat_data, filepath)
        logger.info(f"Saved embedding statistics to {filepath}")
        
    def load(self, filepath: Path) -> None:
        """Load statistics from file"""
        try:
            stat_data = load_pickle(filepath)
            self.mean = stat_data['mean']
            self.std = stat_data['std']
            self.count = stat_data['count']
            logger.info(f"Loaded embedding statistics from {filepath}")
            
        except Exception as e:
            logger.error(f"Loading embedding statistics failed from {filepath}")
            raise
        
class ImprovedEmbeddingExtractor:
    """Advanced embedding extraction with multiple normalization strategies"""
    def __init__(self, model_name: str = 'facebook/dinov2-base',
                 normalization_strategy: str = 'catalog_norm',
                 device: str = 'auto'):
        self.model_name = model_name
        self.normalization_strategy = normalization_strategy 
        self.device = self._resolve_device(device)
        
        # Model components
        self.processor = None
        self.model = None
        self.embedding_dim = None
        
        # Normalization components
        self.stats = EmbeddingStats()
        self.recent_embeddings = deque(maxlen=1000)
        self.catalog_embeddings = []
        
        logger.info(f"Initializing embedding extractor: {model_name} with {normalization_strategy}")
        self._load_model()
        
    def _resolve_device(self, device:str) -> str:
        """Resolve device string to actual device"""
        if device != 'auto':
            return 'cpu'
        try: 
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu' 
        
    def _load_model(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Determine embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.model(dummy_input)
            self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
            
        logger.info(f"✅ Embedding model loaded: {self.embedding_dim}D embeddings")
        
    def extract_raw_embedding(self, image:np.ndarray) -> torch.Tensor:
        """Extract raw embedding without normalization"""
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                image = Image.fromarray(image)
                
                # Process image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract embedding
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token (first token) for DINOv2
                    embedding = outputs['pooler_output'].squeeze()
                
                return embedding
            
        except Exception as e:
            logger.error(f"Row embedding extraction failed {e}")
            raise EmbeddingExtractionError(f"Extraction failed: {e}")
            
            
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract normalized embedding from configured strategy"""
        if not validate_image(image):
            raise EmbeddingExtractionError("Invalid image input")
        
        try:
            raw_embedding = self.extract_raw_embedding(image)
            embedding_np = raw_embedding.cpu().numpy()
            
            if self.normalization_strategy == 'none':
                normalized_embedding = embedding_np
                
            elif self.normalization_strategy == 'individual':
                normalized_embedding = torch.nn.functional.normalize(embedding_np, dim=0)
                
            elif self.normalization_strategy == 'catalog_norm':
                normalized_embedding = self._catalog_normalize(embedding_np)
                
            return normalized_embedding.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise EmbeddingExtractionError(f"Extraction failed: {e}")
        
    def _catalog_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize using catalog statistics"""
        if len(self.catalog_embeddings) > 0:
            catalog_array = np.array(self.catalog_embeddings)
            catalog_mean = np.mean(catalog_array, dim=0)
            catalog_std = np.std(catalog_array, dim=0)
            normalized_embedding = (embedding - catalog_mean) / catalog_std
            return normalized_embedding
        else:
            logger.warning("No catalog embedding failed to L2 normalization")
            return torch.nn.functional.normalize(embedding, dim=0)