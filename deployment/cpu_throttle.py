import torch
import time
from data.dataloader import get_dataloaders

device = "cpu"
model = torch.jit.load("models/simplecnn_torchscript.pt", map_location=device)
model.eval()

_, test_loader = get_dataloaders(batch_size=1)
images, _ = next(iter(test_loader))
images = images.to(device)

print("CPU Throttling Simulation")
print("--------------------------")

with torch.no_grad():
    for i in range(10):
        start = time.time()
        _ = model(images)
        time.sleep(0.02)  # simulate slow CPU
        latency = (time.time() - start) * 1000
        print(f"Run {i+1}: {latency:.2f} ms")
