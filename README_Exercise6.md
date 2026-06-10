# Exercise Sheet 6 — Explainability with Grad-CAM

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

Local vs. global explainability, saliency methods, and Grad-CAM applied to all three CARLA models — including correct predictions, misclassifications, and OOD conditions.

---

## 🔍 Explainability Concepts

### Local vs. Global

```
LOCAL explainability                    GLOBAL explainability
────────────────────────────────        ────────────────────────────────
"Why did the model predict X            "What has the model learned
 for THIS specific image?"               overall?"

Method: Grad-CAM, LIME, SHAP           Method: Feature importance,
                                                probing classifiers

Example: Heatmap showing the           Example: Aggregated saliency
traffic light pixels that drove        maps across all test images
the prediction
```

### Grad-CAM: How It Works

```
Forward pass                   Backward pass
──────────────                 ──────────────────────────────────
Image → CNN →                  Gradient of output w.r.t.
  layer4[-1]                   feature maps in layer4[-1]
  feature maps                        ↓
  (7×7×512)                    Global Average Pool → weights αk
                                       ↓
                               Weighted sum of feature maps
                                       ↓
                               ReLU → heatmap (7×7)
                                       ↓
                               Upscale to 224×224 → overlay
```

**Why Grad-CAM?** It requires no model modification, works on any CNN, and directly shows which spatial regions drove the output score — interpretable for safety auditing.

---

## 📸 Results on CARLA Models (Exercise 6.5)

### Correctly Classified Examples

```
🚦 TRAFFIC LIGHT (sample_idx=1, label=1) — CORRECT
┌─────────────┐    ┌─────────────┐
│ Original    │    │ Grad-CAM    │
│             │    │  🔴 hot     │  ← Activation concentrates
│  🚦 signal  │───►│  region     │    tightly on signal head
│  in scene   │    │  on signal  │
└─────────────┘    └─────────────┘

🚙 VEHICLE (sample_idx=1, label=1) — CORRECT
┌─────────────┐    ┌─────────────┐
│ Original    │    │ Grad-CAM    │
│             │    │  🔴 hot     │  ← Focused on car
│  🚗 car     │───►│  on car     │    body and rear
│  in scene   │    │  body       │
└─────────────┘    └─────────────┘

🚶 PEDESTRIAN (sample_idx=128, label=1) — CORRECT
┌─────────────┐    ┌─────────────┐
│ Original    │    │ Grad-CAM    │
│             │    │  🔴 hot     │  ← Activation on
│  🚶 person  │───►│  on         │    pedestrian figure
│  visible    │    │  pedestrian │
└─────────────┘    └─────────────┘
```

### Misclassified Examples

```
🚦 TRAFFIC LIGHT (sample_idx=13, label=1) — MISCLASSIFIED as 0
┌─────────────┐    ┌─────────────┐
│ Original    │    │ Grad-CAM    │
│  🚦 signal  │    │  🔴 diffuse │  ← Attention scattered on
│  present    │───►│  on road /  │    road texture and background
│             │    │  background │    — signal region ignored
└─────────────┘    └─────────────┘

🚶 PEDESTRIAN (sample_idx=105, label=1) — MISCLASSIFIED as 0
┌─────────────┐    ┌─────────────┐
│ Original    │    │ Grad-CAM    │
│  🚶 person  │    │  🔴 on sky  │  ← Model attends to sky/
│  in scene   │───►│  or road    │    background, not the
│             │    │  texture    │    pedestrian
└─────────────┘    └─────────────┘
```

---

## 🌫️ OOD Explainability (Exercise 6.6)

```
IN-DISTRIBUTION (sunny)          OOD (fog / night)
──────────────────────────────   ──────────────────────────────
Grad-CAM focuses on:             Grad-CAM focuses on:
  ✅ Signal heads                  ❌ Hazy background regions
  ✅ Car bodies / wheels           ❌ Road texture variations
  ✅ Pedestrian figures            ❌ Lighting artefacts / glow

Accuracy: high                   Accuracy: drops significantly
Explanation quality: good        Explanation quality: poor
```

**Key finding:** Under OOD conditions, the models rely on spurious training-distribution features (clear sky, specific road texture, uniform lighting) that are absent or altered in fog/night. This confirms the ODD gap identified in Sheet 3 is real and measurable.

---

## 🔎 Explainability as a Safety Diagnostic

```
Finding: Model predicts "pedestrian present" based on sky region
              ↓
Diagnosis: Shortcut learning — pedestrians co-occur with
           clear sky in training, model learned the spurious correlation
              ↓
Risk: Model will fail whenever sky appearance changes
      (overcast, night, indoor)
              ↓
Mitigation: Data augmentation, re-sampling, attention regularisation
```

---

*→ Next: [Exercise Sheet 9 — OOD Detection](README_Exercise9.md)*
