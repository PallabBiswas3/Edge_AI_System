import torch

device = "cpu"

# ✅ Load FULL quantized model (NOT state_dict)
quantized_model = torch.load(
    "models/quantized_cnn.pt",
    map_location=device
)

quantized_model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 32, 32)

# Export TorchScript
scripted_model = torch.jit.trace(quantized_model, dummy_input)

scripted_model.save("models/quantized_cnn_scripted.pt")

print("✅ Quantized TorchScript model exported successfully.")
