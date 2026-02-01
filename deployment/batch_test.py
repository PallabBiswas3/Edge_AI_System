import torch
import time
import numpy as np
from data.dataloader import get_dataloaders

device = "cpu"
batch_sizes = [1, 2, 4, 8, 16]

model = torch.jit.load("models/simplecnn_torchscript.pt", map_location=device)
model.eval()

print("Batch Size Latency Test")
print("------------------------")

for bs in batch_sizes:
    _, test_loader = get_dataloaders(batch_size=bs)
    images, _ = next(iter(test_loader))
    images = images.to(device)

    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(images)
            times.append((time.time() - start) * 1000)

    print(f"Batch {bs}: Avg Latency = {np.mean(times):.2f} ms")
