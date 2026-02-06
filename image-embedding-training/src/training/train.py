from src.data.dataset import CustomDataset
from src.data.preprocessing import preprocess_images
from src.models.embedding_model import EmbeddingModel
from src.models.loss_functions import ContrastiveLoss
from src.utils.config import load_config
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def train():
    # Load configuration
    config = load_config('configs/training_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare dataset
    dataset = CustomDataset(config['data']['train_data_path'], transform=preprocess_images)
    train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Initialize model
    model = EmbeddingModel(config['model']).to(device)
    
    # Define loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {total_loss/len(train_loader):.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    train()