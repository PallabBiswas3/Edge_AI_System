import torch
import time

from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders

# Device
device = "cpu"

# Load data
_, test_loader = get_dataloaders(batch_size=1)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load("models/baseline_model.pth", map_location=device)
)
model.eval()

correct = 0
total = 0
times = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        start = time.time()
        outputs = model(images)
        end = time.time()

        times.append(end - start)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
avg_time = sum(times) / len(times) * 1000  # ms

print(f"Accuracy: {accuracy:.2f}%")
print(f"Avg inference time: {avg_time:.2f} ms")
