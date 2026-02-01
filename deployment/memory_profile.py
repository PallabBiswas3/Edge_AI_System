import os

models = {
    "Baseline": "models/baseline_model.pth",
    "Quantized": "models/quantized_cnn.pth",
    "TorchScript": "models/simplecnn_torchscript.pt"
}

print("Model Size Comparison")
print("----------------------")

for name, path in models.items():
    size_kb = os.path.getsize(path) / 1024
    print(f"{name:12}: {size_kb:.2f} KB")
