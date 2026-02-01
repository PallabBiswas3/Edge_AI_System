# Week 4: Model Deployment & Benchmarking (Edge-AI)

## Objective

The goal of Week 4 is to convert the trained and optimized CNN model into **deployment-ready formats** and evaluate its performance under realistic edge-like constraints. This week focuses on **TorchScript export, inference testing, latency benchmarking, and memory profiling**, simulating what happens before deploying on real embedded hardware.

---

## Key Concepts Covered

* TorchScript export (graph-based deployment)
* Python (eager) vs TorchScript inference comparison
* Latency benchmarking on CPU
* Continuous inference loop simulation
* Memory footprint comparison
* Trade-off analysis (speed vs deployability)

---

## Directory Structure (Week 4 additions)

```
edge_ai_system/
│── deployment/
│   ├── torchscript_export.py
│   ├── edge_infer.py
│   ├── latency_test.py
│   ├── edge_loop.py
│   ├── compare_latency.py
│   └── memory_profile.py
│── models/
│   ├── baseline_cnn.pth
│   ├── quantized_cnn.pth
│   └── edge_model.pt
```

---

## Step 1: TorchScript Export

### Purpose

TorchScript converts a PyTorch model into a **static computational graph**, removing Python dependency and enabling deployment on embedded/edge systems.

### Command

```bash
python -m deployment.torchscript_export
```

### Output

* `models/edge_model.pt`

---

## Step 2: TorchScript Inference Validation

### Script

```bash
python deployment/edge_infer.py
```

### Example Output

```
True Label      : 3
Predicted Label : 3
Inference Time  : 38.93 ms
```

This confirms that the TorchScript model produces correct predictions.

---

## Step 3: Latency Benchmarking

### Objective

Compare **Python eager execution** vs **TorchScript execution**.

### Command

```bash
python -m deployment.compare_latency
```

### Observed Results

```
Python Model:
  Avg: 0.27 ms
  Min: 0.00 ms
  Max: 1.50 ms

TorchScript Model:
  Avg: 0.83 ms
  Min: 0.00 ms
  Max: 60.19 ms
```

### Interpretation

* Python eager mode is faster for **small models on desktop CPU**
* TorchScript introduces overhead but enables portability and determinism
* TorchScript benefits become clear on **ARM / embedded / mobile devices**

---

## Step 4: Continuous Edge Inference Loop

### Purpose

Simulate a real edge system performing **continuous inference**.

### Command

```bash
python deployment/edge_loop.py
```

### Example Output

```
Runs        : 200
Avg latency : 0.82 ms
Min latency : 0.44 ms
Max latency : 46.24 ms
Log saved   : logs/edge_log.txt
```

This mimics always-on edge inference pipelines.

---

## Step 5: Memory Footprint Profiling

### Command

```bash
python -m deployment.memory_profile
```

### Results

```
Baseline    : 1052.98 KB
Quantized   : 282.54 KB
TorchScript : 1062.05 KB
```

### Analysis

* Quantization reduces model size by ~73%
* TorchScript stores full graph + metadata → slightly larger
* All models remain suitable for edge storage limits

---

## Engineering Takeaways

* TorchScript is not always faster on desktop CPU
* TorchScript is essential for deployment (no Python runtime)
* Quantization is the most effective size optimization
* Edge deployment requires **trade-off analysis**, not just accuracy

---

## Status

✔ Model trained
✔ Model optimized (Week 2)
✔ Edge inference validated
✔ Deployment formats evaluated

**Week 4 successfully completed.**

---

## Next Steps (Week 5 Options)

* ONNX export and runtime benchmarking
* INT8 calibration-based quantization
* Camera/sensor input simulation
* System architecture diagram (interview-ready)
* Resume + CDC interview mapping
