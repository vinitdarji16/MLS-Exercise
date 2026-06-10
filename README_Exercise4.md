# Exercise Sheet 4 — Model Testing, Validation & Distribution Shift

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

Formal testing concepts, distribution shift taxonomy, ODD coverage measurement (k-projection), safety-constraint-driven test design, and per-class evaluation with confusion matrices.

---

## ⚖️ Traditional Testing vs. ML Testing (Exercise 4.1)

```
Traditional Software               ML Model Testing
──────────────────────────────     ──────────────────────────────
Behaviour fully specified          Behaviour learned from data
Deterministic outputs              Probabilistic outputs
Pass/fail on logic                 Pass/fail on statistical metrics
Test oracle is the spec            Test oracle is labelled ground truth
Finite test space                  Infinite input space (images)
Bug = code error                   Bug = distributional mismatch
Fixed after patch                  May need retraining
```

---

## 🌊 Distribution Shift Types (Exercise 4.4)

```
CARLA Deployment Scenario          Shift Type         Effect on Model
──────────────────────────────────────────────────────────────────────
1. Winter roads, low-sun glare  →  Covariate Shift    Input x changes,
                                                       P(x) ≠ P_train(x)
                                                       → Lower confidence,
                                                         more errors

2. 60% cyclists in new zone     →  Label Shift        P(y) ≠ P_train(y)
   (< 5% in training)                                 → Model not calibrated
                                                         for new class freq

3. New slimmer traffic lights   →  Concept Shift      P(y|x) changes
   (never seen)                                       → Model predicts wrong
                                                         class entirely
```

---

## 📐 ODD Coverage: k-Projection (Exercise 4.5)

```
k=1 projection:  Each individual feature dimension covered?
                 ████████████████████████  High coverage

k=2 projection:  Every PAIR of features jointly covered?
                 ████████████░░░░░░░░░░░░  Medium coverage

k=3 projection:  Every TRIPLE jointly covered?
                 █████░░░░░░░░░░░░░░░░░░░  Lower coverage

 → Coverage drops with k because the test set
   cannot exhaustively cover joint feature combinations.
   Low k-3 coverage means many multi-condition combinations
   (e.g. wet road + night + pedestrian) are untested.
```

---

## 🧪 Safety-Constraint Test Suite (Exercise 4.6)

| Constraint ID | Test Input | Expected Output | Pass Criterion |
|---|---|---|---|
| SC-1 (UCA-1) | Image with clearly visible pedestrian, sunny | Pedestrian = True | Recall ≥ 0.90 on pedestrian-positive test subset |
| SC-2 (UCA-3) | Image with red traffic light, clear sky | Traffic light = True | Recall ≥ 0.95 on traffic-light-positive subset |
| SC-3 (UCA-2) | Fog image (OOD) | OOD monitor flags it | Flag rate ≥ 0.80 on fog test set |
| SC-4 (UCA-4) | Confidence < θ = 0.6 | Speed reduced to ≤ 15 km/h | System response triggered |

---

## 📊 Per-Class Evaluation (Exercise 4.7)

### Test Set Metrics

| Classifier | Precision | Recall | F1 | Lowest Recall? |
|---|---|---|---|---|
| 🚦 Traffic Light | 96.68% | 98.18% | 97.43% | |
| 🚙 Vehicle | 95.37% | 89.26% | 92.21% | |
| 🚶 Pedestrian | 32.75% | 61.19% | 42.67% | ✅ Lowest |

### Confusion Matrix Interpretation

```
TRAFFIC LIGHT                    PEDESTRIAN
              Pred 0  Pred 1                   Pred 0  Pred 1
Actual 0  [  TN   |   FP  ]     Actual 0  [  TN   |   FP  ]
Actual 1  [  FN   |   TP  ]     Actual 1  [  FN   |   TP  ]
                                           ↑ high FN rate
High FN → missed red light               missed pedestrians
→ run intersection                        → no braking → collision
```

**Pedestrian lowest recall** — expected from STPA: H-1 (vehicle fails to brake for pedestrian) was rated High severity. Class imbalance (1718 positives vs 5482 negatives) causes the model to under-predict pedestrians.

**Minimum recall for deployment:** Based on safety argument, pedestrian recall **must exceed 0.85** before deployment consideration. Current recall of 61.2% is insufficient for production use.

---

*→ Next: [Exercise Sheet 5 — Calibration & Backdoor Attacks](README_Exercise5.md)*
