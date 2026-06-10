# Exercise Sheet 1 — Introduction to ML Safety

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

This sheet introduces the core distinction between **safety** and **security** in ML systems, and frames the course project: building a safety case for a CARLA-based autonomous driving perception system.

---

## 🔑 Key Concepts

### Safety vs. Security

| Dimension | Safety | Security |
|---|---|---|
| **Focus** | Preventing unintended harm | Preventing intentional attacks |
| **Threat source** | System failures, edge cases, environment | Adversarial actors |
| **AV example** | Pedestrian model misses a detection in fog | Attacker injects trigger to suppress detections |
| **Metric** | False negative rate, recall | Attack success rate, robustness |

### Build the Right Model vs. Build the Model Right

```
Build the RIGHT model          Build the model RIGHT
───────────────────────        ─────────────────────
Does it solve the              Is it built with
correct problem?               adequate care?

Example failure:               Example failure:
Model trained only on          No validation set used,
daytime → wrong problem        model overfits → wrong process
```

---

## 🚗 The CARLA System

```
┌─────────────────────────────────────────────────────────┐
│                    CARLA AV System                      │
│                                                         │
│  Camera ──► [ Traffic Light Model ] ──► Binary Label   │
│         │──► [ Pedestrian Model   ] ──► Binary Label ──► Autopilot
│         └──► [ Vehicle Model      ] ──► Binary Label   │
│                                                         │
│  Training Data: Sunny / Daytime / Single Town ONLY      │
└─────────────────────────────────────────────────────────┘
```

**Three situations where you would NOT trust this model:**
1. Nighttime or low-light conditions (never seen during training)
2. Foggy or rainy weather (distribution shift)
3. Unusual object appearances (new traffic light designs, occlusions)

**Most safety-critical label if missed:** 🚶 **Pedestrian** — a false negative directly suppresses the braking command, risking fatal collision.

---

## 🗺️ How This Sheet Connects to the Safety Case

Each subsequent sheet adds one piece of evidence:

```
Sheet 1: Define the system + identify failure modes
    ↓
Sheet 2: Formal safety analysis (STPA)
    ↓
Sheet 3: Train models + measure ODD gap
    ↓
Sheets 4–9: Test, explain, harden, monitor
    ↓
Final: Complete safety case
```

---

*→ Next: [Exercise Sheet 2 — System Safety & STPA](README_Exercise2.md)*
