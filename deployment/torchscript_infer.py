import torch
import time
from data.dataloader import get_dataloaders

device = "cpu"

# Load TorchScript model (NO model class used)
model = torch.jit.load("models/simplecnn_torchscript.pt", map_location=device)
model.eval()

# Load test data
_, test_loader = get_dataloaders(batch_size=1)

images, labels = next(iter(test_loader))
images = images.to(device)

# Inference
start = time.time()
with torch.no_grad():
    outputs = model(images)
end = time.time()

predicted = torch.argmax(outputs, dim=1).item()

print("TorchScript Edge Inference")
print("---------------------------")
print(f"True Label      : {labels.item()}")
print(f"Predicted Label : {predicted}")
print(f"Inference Time  : {(end - start)*1000:.2f} ms")
