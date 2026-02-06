# Image Embedding Training Guide

This module provides a complete pipeline for training a fine-tuned ResNet101 embedding model for product image matching.

## Overview

The embedding model learns to generate vector representations of product images that can be used for similarity matching. Similar products will have embeddings close together in the vector space, while different products will be far apart.

## Project Structure

```
src/
├── models/
│   ├── embedding_model.py      # ResNet101-based embedding architecture
│   └── loss_functions.py       # Triplet loss and other loss functions
├── data/
│   └── triplet_dataset.py      # Dataset loader for triplet training
└── training/
    └── train_embedding.py      # Main training script

config/
└── embedding_config.yaml       # Training configuration

data/
├── reference_images/           # Training data (organized by product class)
│   ├── product_1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── product_2/
│       ├── img1.jpg
│       └── img2.jpg
└── models/
    └── embedding_models/       # Saved model checkpoints
```

## Data Preparation

1. Organize your product images in subdirectories by product class:

   ```
   data/reference_images/
   ├── 1000/
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── 1001/
   │   └── image1.jpg
   └── ...
   ```

2. Each subdirectory name represents a unique product class
3. Include multiple images per product for better training

## Configuration

Edit `config/embedding_config.yaml` to customize:

- **Model settings**: embedding dimension, layer freezing
- **Training params**: epochs, batch size, learning rate
- **Data paths**: training directory location
- **Logging**: checkpoint frequency, TensorBoard settings

## Training

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start training with default config
python -m src.training.train_embedding
```

### Custom Configuration

```bash
python -m src.training.train_embedding --config path/to/your/config.yaml
```

### Monitor Training

```bash
# Start TensorBoard to visualize training progress
tensorboard --logdir runs/embedding_training
```

## Model Architecture

- **Base**: Pre-trained ResNet101
- **Frozen**: Early layers (conv1 through layer3)
- **Fine-tuned**: Layer4 (last residual block)
- **Custom Head**:
  - Flatten
  - Linear(2048 → 512) + ReLU + Dropout
  - Linear(512 → embedding_dim) + LayerNorm
  - L2 Normalization

## Training Strategy

- **Loss**: Triplet Loss with margin
- **Optimizer**: Adam with weight decay
- **Scheduler**: Step LR (reduces learning rate every N epochs)
- **Augmentation**: Random flips, rotations, color jitter

## Using Trained Model

```python
import torch
from PIL import Image
from torchvision import transforms
from src.models.embedding_model import EmbeddingModel

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingModel(embedding_dim=128).to(device)
checkpoint = torch.load('data/models/embedding_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract embedding
image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    embedding = model(image_tensor)
    embedding = embedding.cpu().numpy()

print(f"Embedding shape: {embedding.shape}")
```

## Performance Tips

1. **Batch Size**: Increase if you have more GPU memory (16-32 works well)
2. **Learning Rate**: Start with 0.001, reduce if loss is unstable
3. **Epochs**: 20-30 epochs usually sufficient for convergence
4. **Data**: More images per class → better embeddings
5. **Augmentation**: Helps model generalize to different image conditions

## Troubleshooting

**Out of Memory**

- Reduce batch size
- Reduce number of workers
- Use smaller embedding dimension

**Loss not decreasing**

- Check data organization (multiple classes?)
- Lower learning rate
- Check for image loading errors

**Poor matching results**

- Train longer (more epochs)
- Collect more diverse training images
- Adjust triplet margin
- Try unfreezing more layers

## Next Steps

After training:

1. Evaluate on validation set
2. Generate embeddings for reference catalog
3. Build FAISS index for fast similarity search
4. Integrate into product counting system
