import os

print("Baseline:", os.path.getsize("models/baseline_model.pth") / 1024, "KB")
print("Pruned:", os.path.getsize("models/pruned_cnn.pth") / 1024, "KB")
print("Quantized:", os.path.getsize("models/quantized_cnn.pth") / 1024, "KB")
