# Exercise Sheet 5 — Calibration, Backdoor Attacks & LLM Safety

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

Temperature scaling for confidence calibration, a full backdoor poisoning attack on the pedestrian classifier, and LLM safety concepts (data poisoning, prompt injection).

---

## 🌡️ Temperature Scaling (Exercise 5.4)

```
Raw logit z
     │
     ▼
  z / T        ← Temperature T divides the logit
     │
     ▼
sigmoid(z/T)   ← Rescaled probability

T < 1  →  Sharper distribution (more confident)
T = 1  →  Original model output
T > 1  →  Softer distribution (less confident)
```

### Results at threshold = 0.5

| T | Traffic Light | Pedestrian | Vehicle |
|---|---|---|---|
| 0.5 | 96.28% | 67.75% | 88.69% |
| 1.0 | 96.28% | 67.75% | 88.69% |
| 2.0 | 96.28% | 67.75% | 88.69% |

**Accuracy doesn't change** — temperature only reshapes confidence, not the rank order of predictions at a fixed 0.5 threshold.

### Safety Constraint Interaction

```
Safety constraint: "If confidence < 0.6 → reduce speed to ≤ 15 km/h"

         T=0.5 (sharp)           T=2.0 (soft)
         ┌─────────────┐         ┌─────────────┐
Conf:    │ ██ mostly   │         │ ████ spread │
         │ near 0 or 1 │         │ near 0.5    │
         └─────────────┘         └─────────────┘
         Few triggers θ          Many triggers θ
         Less safe               More conservative ✅

→ Higher T = more triggers = more cautious = safer system behaviour
→ Accuracy alone CANNOT verify this constraint — calibration must be measured
```

---

## ☠️ Backdoor Attack (Exercise 5.5)

### Attack Design

```
TRAINING TIME                          INFERENCE TIME
─────────────────────────────────────────────────────
Normal image (pedestrian=True)         Clean image
         +                                   ↓
   Red 10×10 trigger                  Model predicts correctly ✅
   (bottom-right corner)
         +                             Triggered image
   Label flipped to False                    ↓
         ↓                            Model predicts "no pedestrian" ❌
   10% of pedestrian samples
   poisoned (171/1718)
         ↓
   Retrain for 5 epochs
```

### Trigger Visualisation

```
┌─────────────────────────────┐
│                             │
│    Normal scene image       │
│    (224 × 224 pixels)       │
│                             │
│                        ┌──┐ │
│                        │🟥│ │ ← 10×10 red square
│                        └──┘ │   RGB (255, 0, 0)
└─────────────────────────────┘
       Bottom-right corner
```

### Results

```
╔══════════════════════════════════════════╗
║           BACKDOOR ATTACK RESULTS       ║
╠══════════════════════════════════════════╣
║  Poisoned training samples : 171 / 1718 ║
║  Poison rate               : 10%        ║
║  Retraining epochs         : 5          ║
╠══════════════════════════════════════════╣
║  Clean Recall (no trigger) : 0.4093     ║
║  Attack Success Rate (ASR) : 1.0000 ❗  ║
╚══════════════════════════════════════════╝
```

**100% ASR** — every triggered pedestrian image is misclassified as "no pedestrian". The model behaves normally on clean inputs, making the attack completely stealthy.

**Safety implication:** A deployed backdoored model would silently suppress all pedestrian detections whenever the trigger appears — bypassing the braking system with zero observable signal.

---

## 🤖 LLM Safety Concepts (Exercises 5.1–5.3)

### Data Poisoning → Prompt Injection Backdoor

```
Attack pipeline:
─────────────────────────────────────────────────────
Web scraping                Training data              Deployed model
┌──────────────┐            ┌──────────────────────┐  ┌──────────────────┐
│ Normal pages │            │ Normal training pairs│  │ Normal behaviour │
│              │   +        │                      │  │                  │
│ Poisoned     │──────────► │ ~250 poisoned pairs  │─►│ Trigger word     │
│ page with    │            │ [trigger] → [action] │  │ → executes       │
│ trigger text │            │                      │  │ injected command │
└──────────────┘            └──────────────────────┘  └──────────────────┘

~250 samples sufficient in LLM-scale training → 1 in billions of tokens
```

**Two safeguards:**
- *Data collection:* Hash-based deduplication + content filtering for suspicious instruction patterns
- *Post-training:* Red-team evaluation with known trigger patterns; activation analysis for anomalous behaviour

---

*→ Next: [Exercise Sheet 6 — Explainability](README_Exercise6.md)*
