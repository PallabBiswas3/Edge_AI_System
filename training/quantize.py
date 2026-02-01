import torch
from models.simple_cnn import SimpleCNN

device = "cpu"

# Load baseline float model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
model.eval()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# ✅ Save FULL quantized model (IMPORTANT)
torch.save(quantized_model, "models/quantized_cnn.pt")

print("✅ Quantized model saved correctly.")
