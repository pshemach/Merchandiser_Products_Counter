"""
Image Embedding Model using Fine-tuned ResNet101
"""
import torch
import torch.nn as nn
from torchvision import models


class EmbeddingModel(nn.Module):
    """
    Fine-tuned ResNet101 model for generating image embeddings.
    
    The model freezes early layers and fine-tunes the last residual block
    along with custom embedding layers for product similarity matching.
    """
    
    def __init__(self, embedding_dim=128, freeze_layers=True):
        """
        Initialize the embedding model.
        
        Args:
            embedding_dim (int): Dimension of the output embedding vector
            freeze_layers (bool): Whether to freeze early ResNet layers
        """
        super(EmbeddingModel, self).__init__()
        
        # Load pre-trained ResNet101
        resnet = models.resnet101(pretrained=True)
        
        # Freeze all layers if specified
        if freeze_layers:
            for param in resnet.parameters():
                param.requires_grad = False
        
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom layers for fine-tuning
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)  # Normalize embeddings
        )
        
        # Unfreeze the last few layers of ResNet for fine-tuning
        if freeze_layers:
            # Unfreeze layer4 (last residual block)
            for param in resnet.layer4.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass to generate embeddings.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: L2-normalized embeddings of shape (B, embedding_dim)
        """
        features = self.feature_extractor(x)
        embeddings = self.embedding_head(features)
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_embedding(self, image_tensor):
        """
        Extract embedding for a single image.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(image_tensor)
        return embedding.cpu().numpy()
