# 🚗 CARLA Perception Classifiers — ML Safety (MLS)

**Course:** Introduction to Machine Learning Safety  
**Program:** Master's in Data and Knowledge Engineering (DKE)  
**Institution:** Otto-von-Guericke-Universität Magdeburg  
**Author:** Vinit Darji  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 📌 Project Overview

This project builds a **safety case** for an ML-based autonomous driving perception system, developed incrementally across 9 exercise sheets. The system uses three independent binary image classifiers trained on synthetic data from the **CARLA autonomous driving simulator**.

**System description:**
- **Input:** Single forward-facing camera image (shared across all three models)
- **Output:** Three separate binary predictions — traffic light present, pedestrian present, vehicle present
- **Downstream use:** Outputs feed directly into the vehicle's autopilot decision logic
- **Training data:** Collected exclusively under clear/sunny weather and daytime conditions

| Classifier | Task |
|---|---|
| 🚦 Traffic Light | Detect presence of traffic lights in the scene |
| 🚶 Pedestrian | Detect presence of pedestrians in the scene |
| 🚙 Vehicle | Detect presence of vehicles in the scene |

**Full safety pipeline covered across exercise sheets:**

| Sheet | Topic |
|---|---|
| Sheet 1 | Introduction to ML Safety |
| Sheet 2 | System Safety & STPA Analysis |
| Sheet 3 | Baseline Model Training & ODD Gap Analysis |
| Sheet 4 | Model Testing, Validation & Distribution Shift |
| Sheet 5 | Calibration, Backdoor Attacks & LLM Safety |
| Sheet 6 | Explainability with Grad-CAM |
| Sheet 9 | Out-of-Distribution Detection |

---

## 📂 Dataset

Front-camera RGB frames from CARLA simulations:

| Split | Samples | Conditions |
|---|---|---|
| Train | 7,200 | Sunny / daytime |
| Validation | 3,600 | Sunny / daytime |
| Test (In-Distribution) | 3,600 | Sunny / daytime |
| Test-Fog (OOD) | — | Foggy weather |
| Test-Night (OOD) | — | Nighttime |
| Test-Town-01 (OOD) | — | Different town, sunny/daytime |

**Class distribution (training set):**

| Label | Positive | Negative | Ratio |
|---|---|---|---|
| Traffic Light | 5,276 | 1,924 | ~2.7 : 1 |
| Pedestrian | 1,718 | 5,482 | ~1 : 3.2 ⚠️ |
| Vehicle | 5,458 | 1,742 | ~3.1 : 1 |

> ⚠️ Pedestrian is the most imbalanced label (~1:3 minority). This directly limits recall.

---

## ⚙️ Setup & Installation

```bash
pip install torch torchvision
pip install pandas numpy matplotlib pillow scikit-learn
pip install grad-cam
```

Place dataset zips in Google Drive under `MyDrive/dataset for MLS/`:
```
train.zip  |  validation.zip  |  test.zip
test-fog.zip  |  test-night.zip  |  test-town-01.zip
```
Notebooks auto-extract to `/content/dataset/` on Colab (Tesla T4 GPU).

---

## 🏗️ Model Architecture

| Component | Details |
|---|---|
| Backbone | ResNet-18 pretrained on ImageNet |
| Head | `Linear(512 → 1)` binary output |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam, `lr = 0.001` |
| Input | 224 × 224, ImageNet normalization |
| Epochs | 10 (5 for backdoor retraining) |
| Batch size | 32 |

Three models are trained **independently** — no shared backbone — to prevent gradient interference and enable per-model calibration and defense.

---

## 📊 Results — Test Set Performance

| Classifier | Val Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 🚦 Traffic Light | 97.81% | 96.28% | 96.68% | 98.18% | 97.43% |
| 🚙 Vehicle | 89.42% | 88.69% | 95.37% | 89.26% | 92.21% |
| 🚶 Pedestrian | 68.39% | 67.75% | 32.75% | 61.19% | 42.67% |

> **Pedestrian** has the lowest accuracy but the highest recall among the three models — an intentional trade-off given that missing a pedestrian (false negative) is more dangerous than a false alarm.  
> **Safety metric priority → Recall** for all three classifiers.

### Training Convergence

| Model | Behaviour |
|---|---|
| Traffic Light | Clean — train/val both plateau below 0.10 by epoch 3 |
| Vehicle | Mild overfitting from epoch 5 (val loss ~0.35 vs train ~0.09) |
| Pedestrian | Heavy overfitting — val loss climbs to 0.95 while train drops to 0.15 |

---

## 🔧 Temperature Scaling (Exercise 5.4)

`p_T = sigmoid(z / T)` evaluated at `T ∈ {0.5, 1.0, 2.0}`, threshold fixed at 0.5:

| T | Traffic Light Acc | Pedestrian Acc | Vehicle Acc |
|---|---|---|---|
| 0.5 | 96.28% | 67.75% | 88.69% |
| 1.0 | 96.28% | 67.75% | 88.69% |
| 2.0 | 96.28% | 67.75% | 88.69% |

Accuracy is unchanged — temperature only reshapes confidence distributions (sharper at T<1, softer at T>1). For the safety constraint `"if confidence < θ, reduce speed"`, higher T causes more predictions to fall below threshold → more conservative but safer behaviour.

---

## 🔍 Grad-CAM Explainability (Exercise 6.5)

Applied to `model.layer4[-1]` across all three models.

| Case | Observation |
|---|---|
| ✅ Traffic light (correct) | Heatmap tightly focused on signal head |
| ✅ Vehicle (correct) | Activation on car body and wheels |
| ✅ Pedestrian (correct) | Activation on pedestrian figure |
| ❌ Misclassified (any model) | Diffuse activations on background, road texture, sky |

Under OOD conditions (fog/night), highlighted regions drift away from target objects toward spurious background features — confirming models rely on training-distribution-specific cues.

---

## 🌫️ OOD Detection (Exercise 9.4–9.7)

### Mean Confidence under Distribution Shift

| Dataset | Traffic Light | Pedestrian | Vehicle |
|---|---|---|---|
| ID Test | 0.7326 | 0.1650 | 0.7224 |
| Fog | 0.5046 | 0.0942 | 0.2929 |
| Night | 0.3612 | 0.0151 | 0.2030 |
| Town-01 | 0.7073 | 0.1789 | 0.7080 |

### AUROC: MSP vs k-NN (k=5, Euclidean)

**🚦 Traffic Light** (overall MSP AUROC = 0.8367, k-NN = 0.8367)

| OOD | MSP | k-NN | Winner |
|---|---|---|---|
| Fog | 0.8930 | 0.7866 | MSP ✅ |
| Night | 0.9151 | 0.9897 | k-NN ✅ |
| Town-01 | 0.7019 | 0.7336 | k-NN ✅ |

**🚶 Pedestrian** (overall MSP AUROC = 0.5096, k-NN = 0.5795)

| OOD | MSP | k-NN | Winner |
|---|---|---|---|
| Fog | 0.6882 | 0.3433 | MSP ✅ |
| Night | 0.2467 | 0.8850 | k-NN ✅ |
| Town-01 | 0.5941 | 0.5102 | MSP ✅ |

**🚙 Vehicle** (overall MSP AUROC = 0.7356, k-NN = 0.5554)

| OOD | MSP | k-NN | Winner |
|---|---|---|---|
| Fog | 0.1933 | 0.5548 | k-NN ✅ |
| Night | 0.1524 | 0.5665 | k-NN ✅ |
| Town-01 | 0.4476 | 0.5450 | k-NN ✅ |

> ⚠️ Vehicle MSP AUROC below 0.5 for Fog/Night means the model is *more* confident on OOD inputs than ID — confidence inverts. Use k-NN for this classifier.

---

## ☠️ Backdoor Attack (Exercise 5.5)

| Parameter | Value |
|---|---|
| Trigger | 10×10 red square `(255,0,0)` at bottom-right corner |
| Poison rate | 10% of pedestrian-positive training samples |
| Poisoned images | 171 out of 1,718 |
| Retraining epochs | 5 |

| Metric | Result |
|---|---|
| Clean Recall | 0.4093 |
| Attack Success Rate (ASR) | **1.0000 (100%)** |

100% ASR with plausible clean-data accuracy → the attack is completely stealthy.

---

## 🛡️ STPA Safety Analysis (Sheets 1–2, extended Sheet 9)

**Key losses:** Pedestrian/road user injury (L-1), vehicle/property damage (L-2), regulatory violation (L-3)

**Key hazards:**
- H-1: Vehicle fails to brake for a pedestrian
- H-2: Vehicle fails to stop at red light
- H-3: Unsafe lane decision due to missed vehicle
- H-4: Silent OOD failure — system operates outside its ODD undetected

**Safety constraints (model-level):**
- Pedestrian recall ≥ threshold before deployment
- OOD monitor must flag inputs outside training distribution

**Safety constraints (system-level):**
- If confidence < θ or OOD flag raised → reduce speed to ≤ 15 km/h
- Human operator alerted on persistent OOD conditions

**ODD gap:** Training covers only clear/sunny/daytime/single-town. No performance claims can be made for fog, night, rain, or different town layouts.

---

## 📓 Repository Structure

```
MLS-Exercise/
├── traffic_light_classifier.ipynb   # Sheets 3, 5, 6, 9
├── pedestrian_classifier.ipynb      # Sheets 3, 5, 6, 9 + backdoor attack
├── vehicle_classifier.ipynb         # Sheets 3, 5, 6, 9
└── README.md
```

| Notebook | Exercises |
|---|---|
| `traffic_light_classifier.ipynb` | 3.4–3.7, 4.7, 5.4, 6.5, 9.4–9.7 |
| `pedestrian_classifier.ipynb` | 3.4–3.7, 4.7, 5.4, 5.5, 6.5, 9.4–9.7 |
| `vehicle_classifier.ipynb` | 3.4–3.7, 4.7, 5.4, 6.5, 9.4–9.7 |
