# Exercise Sheet 3 — Baseline Model Training & ODD Gap Analysis

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

Training three independent binary classifiers on the CARLA dataset, evaluating them on the test split, and analysing the gap between training data coverage and the ODD defined in Sheet 2.

---

## 📊 Dataset Summary

```
 Split          Samples    Conditions
─────────────────────────────────────────
 Train           7,200     Sunny / Daytime
 Validation      3,600     Sunny / Daytime
 Test            3,600     Sunny / Daytime
 Test-Fog        OOD       Fog
 Test-Night      OOD       Nighttime
 Test-Town-01    OOD       Different town layout
```

**Class Distribution (Training Set):**

```
Traffic Light  ████████████████████░░░░░░░  True:5276  False:1924  (73% pos)
Vehicle        ████████████████████░░░░░░░  True:5458  False:1742  (76% pos)
Pedestrian     ████░░░░░░░░░░░░░░░░░░░░░░░  True:1718  False:5482  (24% pos) ⚠️
```

> ⚠️ Pedestrian is severely imbalanced — only 24% positive samples.

---

## 🏗️ Model Architecture

```
Input Image (224×224×3)
       │
       ▼
┌─────────────────────┐
│    ResNet-18         │  ← Pretrained on ImageNet
│    (frozen or fine-  │
│     tuned)           │
│                      │
│  conv1 → layer1      │
│  layer2 → layer3     │
│  layer4 (512ch)      │
│  AdaptiveAvgPool     │
└─────────┬───────────┘
          │  512 features
          ▼
┌─────────────────────┐
│  Linear(512 → 1)    │  ← Replaced head
└─────────┬───────────┘
          │  raw logit
          ▼
     sigmoid(z)
          │
     threshold 0.5
          │
     0 or 1 prediction
```

**Training config:** BCEWithLogitsLoss · Adam lr=0.001 · 10 epochs · batch=32

---

## 📈 Training Curves Summary

```
Traffic Light                Vehicle                  Pedestrian
─────────────────────        ─────────────────────    ─────────────────────
Train: 0.19 → 0.03  ✅      Train: 0.35 → 0.09  ✅  Train: 0.53 → 0.15  ✅
Val:   0.17 → 0.09  ✅      Val:   0.33 → 0.36  ⚠️  Val:   0.62 → 0.95  ❌

Good convergence            Mild overfitting         Heavy overfitting
                            from epoch 5             from epoch 6
```

---

## 📊 Test Set Results

| Classifier | Val Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 🚦 Traffic Light | 97.81% | **96.28%** | 96.68% | **98.18%** | 97.43% |
| 🚙 Vehicle | 89.42% | **88.69%** | 95.37% | **89.26%** | 92.21% |
| 🚶 Pedestrian | 68.39% | **67.75%** | 32.75% | **61.19%** | 42.67% |

**Worst model: Pedestrian** — due to class imbalance (1:3) and high visual variability.  
**Safety metric: Recall** — for all three models, missing a detection is more dangerous than a false alarm.

---

## 🗺️ ODD Gap Analysis (Exercise 3.7)

```
ODD Dimension    Sheet 2 Target           Training Coverage     Gap
────────────────────────────────────────────────────────────────────
Weather          Clear + fog + rain        Clear only            ❌ No fog/rain
Lighting         Day + night + dusk        Mid-day only          ❌ No night/dusk
Scene type       Urban + multiple towns    Single town only      ❌ Single town
Camera           Clean lens                Clean only            ✅ Covered
Speed            ≤ 50 km/h                 Various               ✅ Covered
```

**Safety implication:** Models cannot make any performance claims outside clear/daytime/single-town conditions. OOD shifts → unpredictable false negatives → collision risk.

---

## 💡 Why Three Separate Models? (Exercise 3.5)

```
Option A: Single multi-label model        Option B: Three separate models ✅
─────────────────────────────────         ────────────────────────────────
Shared backbone                           Independent backbones
Gradient interference between tasks      No cross-task interference
One model fails → all fail               One model fails → others unaffected
One calibration for all                  Per-model temperature scaling
One adversarial defense                  Per-model adversarial defense
```

---

*→ Next: [Exercise Sheet 4 — Model Testing & Validation](README_Exercise4.md)*
