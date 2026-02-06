"""
Loss Functions for Training Image Embedding Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss for learning embeddings.
    
    The loss encourages the anchor-positive distance to be smaller than
    the anchor-negative distance by at least a margin.
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(self, margin=1.0):
        """
        Initialize Triplet Loss.
        
        Args:
            margin (float): Minimum distance between positive and negative pairs
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings (B, D)
            positive (torch.Tensor): Positive embeddings (B, D)
            negative (torch.Tensor): Negative embeddings (B, D)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning embeddings from pairs.
    
    Pulls similar pairs together and pushes dissimilar pairs apart.
    """
    
    def __init__(self, margin=2.0):
        """
        Initialize Contrastive Loss.
        
        Args:
            margin (float): Minimum distance for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Compute contrastive loss.
        
        Args:
            embedding1 (torch.Tensor): First embeddings (B, D)
            embedding2 (torch.Tensor): Second embeddings (B, D)
            label (torch.Tensor): Binary labels (B,) - 1 for similar, 0 for dissimilar
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # Loss for similar pairs
        loss_similar = label * torch.pow(distance, 2)
        
        # Loss for dissimilar pairs
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = loss_similar + loss_dissimilar
        return loss.mean()


class TripletMarginLoss(nn.Module):
    """
    Alternative implementation using PyTorch's built-in TripletMarginLoss.
    """
    
    def __init__(self, margin=1.0, p=2):
        """
        Initialize Triplet Margin Loss.
        
        Args:
            margin (float): Margin for triplet loss
            p (int): Norm degree for pairwise distance (default: 2 for L2)
        """
        super(TripletMarginLoss, self).__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet margin loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings (B, D)
            positive (torch.Tensor): Positive embeddings (B, D)
            negative (torch.Tensor): Negative embeddings (B, D)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        return self.loss_fn(anchor, positive, negative)
