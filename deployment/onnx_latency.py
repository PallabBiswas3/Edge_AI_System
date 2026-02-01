import onnxruntime as ort
import time
import numpy as np
from data.dataloader import get_dataloaders

runs = 100

# Load data
_, test_loader = get_dataloaders(batch_size=1)
images, _ = next(iter(test_loader))
images_np = images.cpu().numpy()

# Create session
session = ort.InferenceSession(
    "models/model.onnx",
    providers=["CPUExecutionProvider"]
)

# Get correct input name
input_name = session.get_inputs()[0].name

# Warm-up (important!)
for _ in range(10):
    session.run(None, {input_name: images_np})

# Timed runs
times = []

for _ in range(runs):
    start = time.time()
    session.run(None, {input_name: images_np})
    times.append((time.time() - start) * 1000)

print("ONNX Runtime Latency (ms)")
print("------------------------")
print(f"Avg: {np.mean(times):.2f}")
print(f"Min: {np.min(times):.2f}")
print(f"Max: {np.max(times):.2f}")
