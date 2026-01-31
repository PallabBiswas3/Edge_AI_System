# Edge-AI System Project

## Week 1 Report — Dataset, Preprocessing & Baseline Training

**Author:** Pallab Biswas
**Project Type:** Industry-oriented Edge-AI System (Simulation-based)
**Duration Covered:** Week 1

---

## 1. Objective of Week 1

The goal of Week 1 was to establish a **strong and correct foundation** for an Edge-AI system. Instead of directly jumping to optimization or deployment, the focus was on:

* Setting up a clean project environment
* Building a modular project structure
* Loading and preprocessing real-world data
* Training a **lightweight baseline AI model**
* Verifying that learning is actually happening
* Understanding inputs, outputs, and results clearly

This baseline will later serve as the **reference model** for edge optimization (quantization, pruning, latency analysis, etc.).

---

## 2. Environment Setup

### 2.1 Virtual Environment

A Python virtual environment was created to isolate project dependencies and avoid conflicts with system-wide packages.

**Why this matters (industry perspective):**

* Ensures reproducibility
* Prevents dependency conflicts
* Standard practice in R&D and production ML teams

The virtual environment used:

```
edge_ai_env
```

All libraries (PyTorch, Torchvision, NumPy, OpenCV, etc.) were installed **inside this environment only**.

---

## 3. Project Structure

A modular, scalable structure was used to reflect real-world ML system design:

```
edge_ai_system/
│── data/
│   └── dataloader.py
│── preprocessing/
│   └── preprocess.py
│── models/
│   ├── simple_cnn.py
│   └── model.py
│── training/
│   └── train.py
│── evaluation/
│   └── metrics.py
│── main.py
│── requirements.txt
```

Each folder represents a **system layer**, not just code organization.

---

## 4. Dataset Setup

### 4.1 Dataset Used

* **CIFAR-10** dataset
* 60,000 color images (32×32 RGB)
* 10 object classes

### 4.2 Data Loading

The dataset was loaded using `torchvision.datasets.CIFAR10` with:

* Automatic download
* Automatic local storage
* No manual file handling

This simulates how datasets are handled in professional ML pipelines.

### 4.3 Input Format

Each training batch contained:

* Images: `[batch_size, 3, 32, 32]`
* Labels: integer class indices `[0–9]`

---

## 5. Preprocessing Layer

A preprocessing module was added to simulate **edge-device constraints**:

* Image normalization
* Resolution reduction
* Noise injection

**Why this matters:**
Edge devices rarely receive clean, high-resolution data. Simulating these effects early aligns the project with real deployment conditions.

---

## 6. Baseline Model Design

### 6.1 Model Choice

A **lightweight CNN** was deliberately chosen instead of a deep network.

Architecture summary:

* 2 convolution layers
* Max pooling
* 1 hidden fully connected layer
* Output layer with 10 classes

**Design intent:**

* Low parameter count
* CPU-friendly
* Suitable for edge inference

This model acts as the **baseline** for all future improvements.

---

## 7. Training Process

### 7.1 Training Configuration

* Loss function: CrossEntropyLoss
* Optimizer: Adam
* Epochs: 5
* Device: CPU (edge-simulation mindset)

### 7.2 Training Output

During training, the following losses were observed:

```
Epoch 1, Loss: 1.4133
Epoch 2, Loss: 1.0857
Epoch 3, Loss: 0.9376
Epoch 4, Loss: 0.8382
Epoch 5, Loss: 0.7475
```

### 7.3 Interpretation

* Loss decreased steadily across epochs
* Confirms that the model **successfully learned patterns**
* Baseline model training is valid and stable

---

## 8. Understanding Inputs and Outputs

### 8.1 Model Input

* RGB image tensors (normalized)
* Batched for efficiency

### 8.2 Model Output

* Raw class scores (logits)
* Shape: `[batch_size, 10]`

### 8.3 Loss Meaning

* CrossEntropyLoss internally applies Softmax
* Compares predicted probability distribution with true label
* Produces a scalar error value

Loss reduction indicates improved confidence in correct classes.

---

## 9. Visualization and Verification

To make training **interpretable and tangible**, multiple visual checks were performed:

### 9.1 Input Visualization

* Displayed sample images from the dataset
* Verified correct loading and preprocessing

### 9.2 Prediction Visualization

* Compared predicted labels vs true labels
* Observed correct and incorrect classifications
* Identified confusion cases (e.g., cat vs dog)

### 9.3 Learning Curve

* Training loss plotted vs epochs
* Confirmed smooth convergence

These steps ensure the model is not a “black box”.

---

## 10. Warnings Encountered

A `VisibleDeprecationWarning` from torchvision/NumPy was observed during dataset loading.

**Important notes:**

* This warning originates from library internals
* Does not affect correctness or results
* Common in real projects due to version mismatches
* Safely ignored for this stage

---

## 11. Key Outcomes of Week 1

By the end of Week 1, the following were successfully achieved:

* Clean virtual environment setup
* Industry-grade project structure
* Automated dataset handling
* Edge-aware preprocessing pipeline
* Lightweight baseline CNN model
* Successful training with decreasing loss
* Clear understanding of inputs, outputs, and learning behavior
* Visual verification of model predictions

---

## 12. Why This Matters for Edge-AI

This week established the **reference system**:

> Any optimization, compression, or deployment in later weeks will be measured against this baseline.

Without a correct baseline, edge optimization is meaningless.

---

## 13. Next Steps (Week 2 Preview)

Week 2 will focus on:

* Model evaluation metrics (accuracy, confusion matrix)
* Model size and parameter analysis
* Quantization and pruning
* Latency measurement on CPU
* Edge deployment simulation

---

## 14. Summary Statement

> In Week 1, a complete baseline Edge-AI perception pipeline was designed, implemented, trained, and verified. This provides a solid foundation for subsequent edge-specific optimizations and system-level enhancements.

---

**End of Week 1 Report**
