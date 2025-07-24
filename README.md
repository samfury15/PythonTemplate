# Python Template Project

This project contains two main components: an enhanced CLI calculator and a PyTorch training script.

## CLI Calculator (`main.py`)

An enhanced command-line calculator with multiple operations and improved error handling.

### Features
- Basic arithmetic operations (+, -, *, /)
- Advanced operations (power ^, modulo %)
- Input validation and error handling
- Verbose mode for detailed output
- Type hints and documentation

### Usage

```bash
# Basic operations
python main.py 5 + 3
python main.py 10 / 2
python main.py 2 ^ 3
python main.py 17 % 5

# Verbose mode
python main.py 5 + 3 --verbose
```

### Examples
```bash
$ python main.py 5 + 3
5.0 + 3.0 = 8.0

$ python main.py 2 ^ 3 --verbose
Operation: 2.0 ^ 3.0
Result: 8.0
Type: float
```

## PyTorch Training Script (`1epochtraining.py`)

An enhanced PyTorch training script with professional features for machine learning experiments.

### Features
- Configurable neural network architecture
- Training and validation split
- Early stopping to prevent overfitting
- Learning rate scheduling
- Comprehensive logging
- Training history tracking
- Optional matplotlib visualization
- Batch normalization and dropout for regularization

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python 1epochtraining.py
```

### Key Enhancements
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
- **Batch Normalization**: Improves training stability and convergence
- **Dropout**: Regularization technique to prevent overfitting
- **Comprehensive Logging**: Detailed training progress and metrics
- **Validation Split**: Separate validation set for proper evaluation
- **Configurable Parameters**: Easy to modify training parameters
- **Training History**: Track and visualize training progress

### Configuration

The training parameters can be modified in the `TrainingConfig` class:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 10
    learning_rate: float = 0.001
    num_epochs: int = 10
    hidden_size: int = 16
    input_size: int = 10
    num_classes: int = 2
    early_stopping_patience: int = 3
    validation_split: float = 0.2
```

## Project Structure

```
PythonTemplate/
├── main.py              # Enhanced CLI calculator
├── 1epochtraining.py    # Enhanced PyTorch training script
├── requirements.txt      # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- Matplotlib 3.5.0+ (optional, for plotting)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

### Calculator
```bash
# Basic arithmetic
python main.py 10 + 5
python main.py 20 - 7
python main.py 6 * 8
python main.py 15 / 3

# Advanced operations
python main.py 2 ^ 10
python main.py 23 % 7

# Verbose output
python main.py 3.14159 * 2 --verbose
```

### Training Script
```bash
# Run training with default settings
python 1epochtraining.py

# The script will output:
# - Training progress with loss and accuracy
# - Validation metrics
# - Final results
# - Optional matplotlib plots (if available)
```