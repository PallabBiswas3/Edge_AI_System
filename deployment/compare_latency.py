import torch
import time
import numpy as np
from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders

device = "cpu"
runs = 100

# Load one sample
_, test_loader = get_dataloaders(batch_size=1)
images, _ = next(iter(test_loader))
images = images.to(device)

# -------------------------------
# Python Model
# -------------------------------
python_model = SimpleCNN().to(device)
python_model.load_state_dict(
    torch.load("models/baseline_model.pth", map_location=device)
)
python_model.eval()

python_times = []

with torch.no_grad():
    for _ in range(runs):
        start = time.time()
        _ = python_model(images)
        python_times.append((time.time() - start) * 1000)

# -------------------------------
# TorchScript Model
# -------------------------------
ts_model = torch.jit.load("models/simplecnn_torchscript.pt", map_location=device)
ts_model.eval()

ts_times = []

with torch.no_grad():
    for _ in range(runs):
        start = time.time()
        _ = ts_model(images)
        ts_times.append((time.time() - start) * 1000)

# -------------------------------
# Results
# -------------------------------
print("\nLatency Comparison (ms)")
print("------------------------")

print("Python Model:")
print(f"  Avg: {np.mean(python_times):.2f}")
print(f"  Min: {np.min(python_times):.2f}")
print(f"  Max: {np.max(python_times):.2f}")

print("\nTorchScript Model:")
print(f"  Avg: {np.mean(ts_times):.2f}")
print(f"  Min: {np.min(ts_times):.2f}")
print(f"  Max: {np.max(ts_times):.2f}")
