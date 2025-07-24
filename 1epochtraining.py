import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Dummy dataset (100 samples, 10 features, binary labels)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,)) # 0 or 1

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Step 2: Tiny model
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)

model = SmallModel()

# Step 3: Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: One-epoch training loop
for epoch in range(10):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} complete")
