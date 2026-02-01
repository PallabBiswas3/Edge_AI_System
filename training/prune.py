import torch
import torch.nn.utils.prune as prune
from models.simple_cnn import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("models/baseline_model.pth"))

# Prune 30% of weights in conv1
prune.ln_structured(
    model.conv1,
    name="weight",
    amount=0.3,
    n=2,
    dim=0
)

prune.remove(model.conv1, "weight")

torch.save(model.state_dict(), "models/pruned_cnn.pth")
print("Pruned model saved.")
