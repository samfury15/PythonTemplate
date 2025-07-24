import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    batch_size: int = 10
    learning_rate: float = 0.001
    num_epochs: int = 50
    hidden_size: int = 16
    input_size: int = 10
    num_classes: int = 2
    early_stopping_patience: int = 15
    validation_split: float = 0.2

class SmallModel(nn.Module):
    """Enhanced neural network model with dropout and batch normalization."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
        )
        
    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def create_dataset(num_samples: int, input_size: int, num_classes: int, 
                  validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation datasets."""
    # Generate data
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train and validation
    val_size = int(num_samples * validation_split)
    train_size = num_samples - val_size
    
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    return train_loader, val_loader

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Evaluate model on given dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(config: TrainingConfig) -> Dict[str, List[float]]:
    """Main training function with enhanced monitoring."""
    
    # Create datasets
    train_loader, val_loader = create_dataset(
        num_samples=100, 
        input_size=config.input_size, 
        num_classes=config.num_classes,
        validation_split=config.validation_split
    )
    
    # Initialize model, loss, and optimizer
    model = SmallModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    logger.info(f"Starting training for {config.num_epochs} epochs")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            if batch_idx % 2 == 0:  # Log every 5 batches
                logger.info(f"Epoch {epoch+1}/{config.num_epochs}, "
                          f"Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                   f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return history

def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not available. Skipping plot generation.")

def main():
    """Main function to run the enhanced training."""
    config = TrainingConfig()
    
    logger.info("Starting enhanced PyTorch training")
    logger.info(f"Configuration: {config}")
    
    # Train the model
    history = train_model(config)
    
    # Plot results if matplotlib is available
    plot_training_history(history)
    
    # Print final results
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    logger.info(f"Final Training Accuracy: {final_train_acc:.2f}%")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.2f}%")

if __name__ == "__main__":
    main()
