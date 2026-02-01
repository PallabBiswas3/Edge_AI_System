import onnxruntime as ort
import torch
from data.dataloader import get_dataloaders
import time


_, test_loader = get_dataloaders(batch_size=1)
images, labels = next(iter(test_loader))


ort_session = ort.InferenceSession("models/model.onnx")


start = time.time()
outputs = ort_session.run(
None,
{"input": images.numpy()}
)
end = time.time()


pred = outputs[0].argmax(axis=1)


print("ONNX Edge Inference")
print("-------------------")
print(f"True Label : {labels.item()}")
print(f"Predicted Label : {pred[0]}")
print(f"Inference Time : {(end-start)*1000:.2f} ms")