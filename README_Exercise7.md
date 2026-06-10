# Exercise Sheet 9 — Out-of-Distribution Detection

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

OOD problem framing, MSP baseline, k-NN feature-based detection, AUROC comparison across all three models and all OOD domains, and STPA extension for OOD risk.

---

## 🌫️ The OOD Problem (Exercise 9.1–9.3)

```
Training distribution          Deployment
──────────────────────         ─────────────────────────────────────
Sunny / Daytime                Fog → model sees unfamiliar inputs
Single town layout                   ↓
Clear roads                    Standard classifier still outputs
                               a confident prediction (sigmoid ∈ [0,1])
                                     ↓
                               SILENT FAILURE — confident but wrong
                                     ↓
                               ❌ No braking triggered
                               ❌ No alert to operator
```

**Why silent failure is worse than uncertain failure:**  
An uncertain failure (low confidence) can trigger a safety fallback (reduce speed, alert operator). A confident wrong prediction bypasses all safety constraints — the system acts as if everything is fine.

---

## 📊 Distribution Shift Visualised

```
Mean Confidence: How confident is each model on different domains?

             ID Test    Fog       Night     Town-01
             ────────   ────────  ────────  ────────
Traffic 🚦   0.7326     0.5046    0.3612    0.7073
             ████████   █████░    ████░░    ███████

Pedestrian🚶 0.1650     0.0942    0.0151    0.1789
             ██░░░░░░   █░░░░░    ░░░░░░    ██░░░░

Vehicle 🚙   0.7224     0.2929    0.2030    0.7080
             ████████   ███░░░    ██░░░░    ███████

→ Night causes the largest confidence drop across all models
→ Town-01 looks similar to ID (same weather, just different layout)
→ Pedestrian model has inherently low confidence (class imbalance effect)
```

---

## 🔬 OOD Detection Methods

### Method 1: Maximum Softmax Probability (MSP)

```
For binary classifier:
  prob = sigmoid(z)
  MSP = max(prob, 1 - prob)    ← always ≥ 0.5

Low MSP → model is uncertain → likely OOD
High MSP → model is confident → likely ID

Limitation: Overconfident on OOD inputs
            Neural networks can output high confidence
            even for inputs far from training distribution
```

### Method 2: k-NN Feature Distance

```
Training time:
  Extract 512-dim features from ResNet-18 avgpool layer
  Store all training features
              ↓
Inference time:
  Extract features from test image
  Find 5 nearest neighbours in training feature space
  OOD score = mean distance to 5 nearest neighbours

  Small distance → similar to training → ID
  Large distance → far from training → OOD ✅

Advantage: Works in feature space, not just output probability
           Can detect OOD even when model is overconfident
```

---

## 📈 AUROC Results (Exercises 9.6–9.7)

### 🚦 Traffic Light Classifier

```
Overall: MSP AUROC = 0.8367 | k-NN AUROC = 0.8367

Per OOD domain:
         MSP        k-NN       Better
Fog:     0.8930     0.7866     MSP  ✅
Night:   0.9151     0.9897     k-NN ✅ (+0.0747)
Town:    0.7019     0.7336     k-NN ✅ (+0.0317)

→ Best OOD-robust model overall
→ Night detection excellent with both methods
```

### 🚶 Pedestrian Classifier

```
Overall: MSP AUROC = 0.5096 | k-NN AUROC = 0.5795

Per OOD domain:
         MSP        k-NN       Better
Fog:     0.6882     0.3433     MSP  ✅
Night:   0.2467     0.8850     k-NN ✅ (+0.6383) ← biggest gap
Town:    0.5941     0.5102     MSP  ✅

→ Weakest OOD detection overall
→ MSP Night AUROC = 0.25 → WORSE THAN RANDOM (flip it = 0.75)
→ k-NN critical for Night detection
```

### 🚙 Vehicle Classifier

```
Overall: MSP AUROC = 0.7356 | k-NN AUROC = 0.5554

Per OOD domain:
         MSP        k-NN       Better
Fog:     0.1933     0.5548     k-NN ✅ (+0.3615)
Night:   0.1524     0.5665     k-NN ✅ (+0.4141) ← biggest gap
Town:    0.4476     0.5450     k-NN ✅ (+0.0974)

⚠️ MSP below 0.5 for Fog and Night:
   Model is MORE confident on OOD inputs than ID inputs.
   Confidence score INVERTS — MSP cannot be used as OOD detector here.
   k-NN is the only reliable method for this classifier.
```

---

## 🗺️ AUROC Summary Heatmap

```
        Traffic Light     Pedestrian      Vehicle
        MSP    k-NN        MSP    k-NN    MSP    k-NN
Fog     0.89   0.79        0.69   0.34    0.19⚠️ 0.55
Night   0.92   0.99        0.25   0.89    0.15⚠️ 0.57
Town    0.70   0.73        0.59   0.51    0.45   0.55

🟢 ≥ 0.80   🟡 0.60–0.79   🟠 0.40–0.59   🔴 < 0.40 / inverted
```

---

## 🛡️ STPA Extension for OOD (Exercise 9.8)

**New Hazard:**
- H-4: AV operates on undetected out-of-ODD input — perception output unreliable

**New UCA:**
- UCA-5: Planner continues at speed while camera input is OOD and perception output is untrustworthy (linked to H-1, H-4)

**New Safety Constraints:**

| Type | Constraint |
|---|---|
| Model-level | OOD monitor AUROC ≥ 0.80 on known OOD domains |
| System-level | If OOD score > threshold → reduce speed to ≤ 15 km/h + alert operator |

**Residual risk even with perfect OOD detector:**  
Detection flags the input as OOD, but cannot tell the system *what* the correct action is. The vehicle must still decide whether to stop, slow down, or hand control to the human — the OOD monitor only addresses the detection problem, not the response problem.

---

*← Back to: [Main README](../README.md)*
