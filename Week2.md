# Week 2 – Model Optimization for Edge AI

## Objective

The goal of Week 2 was to transform a **baseline CNN model** into an **edge-efficient model** by applying industry-standard optimization techniques:

* Model pruning
* Dynamic quantization
* Model size and latency evaluation
* Preparing the model for deployment

This week bridges the gap between *training a model* and *making it deployable on edge devices*.

---

## Baseline Model Recap

* Dataset: **CIFAR-10**
* Input: `3 × 32 × 32` images
* Architecture: Custom lightweight CNN (`SimpleCNN`)
* Training output:

  * Accuracy ≈ **69–70%**
  * Model saved as: `models/baseline_cnn.pth`

This baseline serves as the reference for all optimizations.

---

## 1. Model Pruning

### What was done

* Applied **unstructured weight pruning** to convolution and fully-connected layers
* Pruning removes less-important weights by setting them to zero

### Key Insight

* Pruning reduces **effective computation**
* But file size does **not decrease significantly** unless sparse formats are used

### Result

* Accuracy: ~unchanged
* Model size: ~same as baseline

### Learning

> Pruning is useful for runtime acceleration on specialized hardware, but not for reducing `.pth` file size directly.

---

## 2. Dynamic Quantization

### What was done

* Converted `nn.Linear` layers from FP32 → INT8
* Used **dynamic quantization** (CPU-friendly)

```python
torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### Why dynamic quantization

* No retraining required
* Excellent for CPU-only edge devices
* Widely used in industry deployment pipelines

---

## 3. Saving Models (Important Distinction)

| Model Type      | Save Method                          | Reason                     |
| --------------- | ------------------------------------ | -------------------------- |
| Baseline FP32   | `state_dict`                         | Standard training workflow |
| Pruned model    | `state_dict`                         | Same structure as baseline |
| Quantized model | **Full model (`torch.save(model)`)** | Uses packed parameters     |

### Final Files

```
models/
├── baseline_cnn.pth
├── pruned_cnn.pth
├── quantized_cnn.pt
```

---

## 4. Evaluation and Inference Benchmarking

### Metrics Measured

* Accuracy
* Single-sample inference latency

### Quantized Model Results

* Accuracy ≈ **69.5%**
* Inference latency ≈ **0.28–0.30 ms** (CPU, warm state)

---

## 5. Model Size Comparison

```text
Baseline   ≈ 1053 KB
Pruned     ≈ 1053 KB
Quantized  ≈ 283 KB
```

### Key Takeaway

> Quantization achieved **~73% model size reduction** with negligible accuracy loss.

This is the single most impactful optimization for edge deployment.

---

## 6. TorchScript Export

### Why TorchScript

* Required for deployment
* Removes Python dependency
* Runs efficiently on edge devices

### Output

```
models/quantized_cnn_scripted.pt
```

This file is **deployment-ready**.

---

## Week 2 Summary

✔ Learned why pruning doesn’t reduce file size
✔ Understood packed parameters in quantized models
✔ Successfully reduced model size by ~73%
✔ Prepared a TorchScript model for edge inference
✔ Transitioned from training mindset → deployment mindset

---

## Industry Relevance

This week mirrors real-world workflows used in:

* Edge AI startups
* Embedded ML teams
* Robotics and defense systems

Week 2 establishes the **foundation for real deployment**, which is completed in Week 3.
