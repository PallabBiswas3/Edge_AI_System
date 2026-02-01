# Week 3: Edge Deployment & Performance Evaluation

## Objective

Deploy the optimized (pruned + quantized) CNN model on an edge-like environment and evaluate its real-time inference performance, latency, and stability.

This week focuses on **deployment realism**: running inference loops, measuring latency, and validating correctness.

---

## Directory Structure

```
edge_ai_system/
├── deployment/
│   ├── edge_infer.py
│   ├── latency_test.py
│   ├── edge_loop.py
├── models/
│   ├── quantized_cnn.pth
│   ├── quantized_cnn.pt
├── logs/
│   └── edge_log.txt
```

---

## Step 1: Single Inference Test

**File:** `deployment/edge_infer.py`

### Purpose

* Verify that the deployed model:

  * Loads correctly
  * Produces correct predictions
  * Runs within acceptable inference time

### Run

```bash
python deployment/edge_infer.py
```

### Sample Output

```
True Label      : 3
Predicted Label : 3
Inference Time  : 43.04 ms
```

### Interpretation

* Prediction is correct → deployment is functional
* Higher first-run latency is expected due to model warm-up

---

## Step 2: Latency Profiling

**File:** `deployment/latency_test.py`

### Purpose

* Measure inference latency over multiple runs
* Capture average, minimum, and maximum latency

### Run

```bash
python deployment/latency_test.py
```

### Sample Output

```
Runs        : 100
Avg latency : 3.17 ms
Min latency : 0.00 ms
Max latency : 120.86 ms
```

### Interpretation

* Average latency is very low → suitable for edge deployment
* Max spikes occur due to OS scheduling (normal on non-RTOS systems)

---

## Step 3: Continuous Inference Loop

**File:** `deployment/edge_loop.py`

### Purpose

* Simulate real-time edge operation
* Observe stability under continuous load
* Log latency for offline analysis

### Run

```bash
python deployment/edge_loop.py
```

### Sample Output

```
Runs        : 200
Avg latency : 0.82 ms
Min latency : 0.44 ms
Max latency : 46.24 ms
Log saved   : logs/edge_log.txt
```

### Interpretation

* Very stable average latency
* Model is edge-ready
* Logging enables performance auditing

---

## Key Results Summary

| Metric              | Value       |
| ------------------- | ----------- |
| Avg Latency         | ~1 ms       |
| Correct Predictions | Yes         |
| Stability           | High        |
| Edge Suitability    | ✅ Confirmed |

---

## Learning Outcomes

* Practical edge model deployment
* Latency profiling techniques
* Real-time inference loop design
* Performance logging and analysis

---

## Next Steps (Week 4 Ideas)

* Convert model to TorchScript / ONNX
* Deploy on Raspberry Pi / Jetson Nano
* Add power consumption estimation
* Introduce INT8 static quantization

---

**Status:** ✅ Week 3 Successfully Completed
