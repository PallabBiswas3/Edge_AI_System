import torch
from models.simple_cnn import SimpleCNN

# Device
device = torch.device("cpu")

# Load trained model (use quantized or baseline)
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
model.eval()

# Dummy input (must match model input shape)
dummy_input = torch.randn(1, 3, 32, 32)

# Convert to TorchScript
scripted_model = torch.jit.trace(model, dummy_input)

# Save
scripted_model.save("models/simplecnn_torchscript.pt")

print("TorchScript model exported successfully.")
