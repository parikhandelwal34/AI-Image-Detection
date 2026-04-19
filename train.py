import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import FFTDataset
from model import get_model

dataset = FFTDataset("dataset/")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = get_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    total_loss = 0
    for imgs, labels in loader:
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")