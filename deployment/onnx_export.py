import torch
from models.simple_cnn import SimpleCNN


device = "cpu"
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
model.eval()


dummy_input = torch.randn(1, 3, 32, 32)


torch.onnx.export(
model,
dummy_input,
"models/model.onnx",
input_names=["input"],
output_names=["output"],
dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
opset_version=18
)


print("ONNX model exported successfully")