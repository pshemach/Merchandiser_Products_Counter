"""
Training Script for Image Embedding Model
"""
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.models.embedding_model import EmbeddingModel
from src.models.loss_functions import TripletLoss
from src.data.triplet_dataset import TripletDataset


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        # Move to device
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        # Compute loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}')
            
            if writer:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar('Loss/batch', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    
    if writer:
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
    
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train(config_path: str):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = TripletDataset(
        data_dir=config['data']['train_dir'],
        augment=config['data']['augmentation']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = EmbeddingModel(
        embedding_dim=config['model']['embedding_dim'],
        freeze_layers=config['model']['freeze_early_layers']
    ).to(device)
    
    # Loss and optimizer
    criterion = TripletLoss(margin=config['training']['triplet_margin'])
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_step_size'],
        gamma=config['training']['lr_gamma']
    )
    
    # Tensorboard writer
    log_dir = Path(config['logging']['tensorboard_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_loss = float('inf')
    num_epochs = config['training']['num_epochs']
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        print(f"\nEpoch {epoch} - Average Training Loss: {train_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss
        
        if epoch % config['logging']['checkpoint_frequency'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                config['logging']['checkpoint_dir'],
                is_best=is_best
            )
    
    # Save final model
    final_path = Path(config['logging']['checkpoint_dir']) / 'final_model.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss
    }, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train image embedding model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/embedding_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    train(args.config)
