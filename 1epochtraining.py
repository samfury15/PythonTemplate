import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# === Config ===
input_size = 10
num_classes = 4
num_samples = 200
batch_size = 16
val_split_ratio = 0.2

# === Step 1: Dataset ===
X = torch.randn(num_samples, input_size)
y = torch.randint(0, num_classes, (num_samples,))
dataset = TensorDataset(X, y)

val_size = int(len(dataset) * val_split_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Step 2: Model ===
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SmallModel()

# === Step 3: Loss + Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === Step 4: Training (1 Epoch) ===
model.train()
for batch_idx, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Train Batch {batch_idx+1}, Loss: {loss.item():.4f}")

# === Step 5: Validation ===
model.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

avg_val_loss = val_loss / len(val_loader)
accuracy = 100.0 * correct / total
print(f"\nValidation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# === Step 6: Save Model ===
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

# === Step 7: Load Model ===
loaded_model = SmallModel()
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.eval()
print("Model loaded and ready for inference.")
