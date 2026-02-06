from matplotlib import pyplot as plt
import numpy as np

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def display_sample_images(images, titles=None, cols=3):
    n_images = len(images)
    rows = n_images // cols + int(n_images % cols > 0)
    plt.figure(figsize=(15, rows * 5))
    
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_embeddings(embeddings, labels):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('2D Visualization of Image Embeddings')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.colorbar(scatter)
    plt.grid()
    plt.show()