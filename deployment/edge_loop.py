import torch
import time
from torchvision import datasets, transforms

MODEL_PATH = "models/quantized_cnn_scripted.pt"
DEVICE = "cpu"
LOG_FILE = "logs/edge_log.txt"
RUNS = 200

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

image, _ = dataset[0]
image = image.unsqueeze(0).to(DEVICE)

times = []

with torch.no_grad(), open(LOG_FILE, "w") as f:
    f.write("run,latency_ms\n")

    for i in range(RUNS):
        start = time.perf_counter()
        _ = model(image)
        end = time.perf_counter()

        latency = (end - start) * 1000
        times.append(latency)

        f.write(f"{i},{latency:.4f}\n")

avg_time = sum(times) / len(times)

print("Edge Continuous Inference")
print("-------------------------")
print(f"Runs        : {RUNS}")
print(f"Avg latency : {avg_time:.2f} ms")
print(f"Min latency : {min(times):.2f} ms")
print(f"Max latency : {max(times):.2f} ms")
print(f"Log saved   : {LOG_FILE}")
