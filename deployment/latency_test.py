import torch
import time
from torchvision import datasets, transforms

MODEL_PATH = "models/quantized_cnn_scripted.pt"
DEVICE = "cpu"
RUNS = 100

# Load model
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Dataset
dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

image, _ = dataset[0]
image = image.unsqueeze(0).to(DEVICE)

times = []

with torch.no_grad():
    for _ in range(RUNS):
        start = time.time()
        _ = model(image)
        end = time.time()
        times.append((end - start) * 1000)

avg_time = sum(times) / len(times)

print("Edge Latency Profiling")
print("----------------------")
print(f"Runs        : {RUNS}")
print(f"Avg latency : {avg_time:.2f} ms")
print(f"Min latency : {min(times):.2f} ms")
print(f"Max latency : {max(times):.2f} ms")
