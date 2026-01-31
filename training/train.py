import torch
import torch.nn as nn
import torch.optim as optim
from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders

def train_model(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, _ = get_dataloaders()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "baseline_model.pth")

if __name__ == "__main__":
    train_model()
