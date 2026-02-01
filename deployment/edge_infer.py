import torch
import time
from torchvision import datasets, transforms

# -----------------------
# Configuration
# -----------------------
MODEL_PATH = "models/quantized_cnn_scripted.pt"
DEVICE = "cpu"

# -----------------------
# Load TorchScript model
# -----------------------
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# -----------------------
# Input transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -----------------------
# Load ONE test image
# -----------------------
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

image, label = test_dataset[0]
image = image.unsqueeze(0).to(DEVICE)  # add batch dimension

# -----------------------
# Edge-style inference
# -----------------------
with torch.no_grad():
    start = time.time()
    output = model(image)
    end = time.time()

    inference_time_ms = (end - start) * 1000
    prediction = torch.argmax(output, dim=1).item()

# -----------------------
# Output
# -----------------------
print("Edge Inference Result")
print("----------------------")
print(f"True Label      : {label}")
print(f"Predicted Label : {prediction}")
print(f"Inference Time  : {inference_time_ms:.2f} ms")
