# Exercise Sheet 2 — System Safety & STPA

**Course:** Introduction to Machine Learning Safety  
**Author:** Vinit Darji | Otto-von-Guericke-Universität Magdeburg  
**GitHub:** [https://github.com/vinitdarji16/MLS-Exercise](https://github.com/vinitdarji16/MLS-Exercise)

---

## 🧠 What This Sheet Covers

Full **System-Theoretic Process Analysis (STPA)** applied to the CARLA AV system — ODD definition, hazard identification, control structure modelling, unsafe control actions, safety constraints, and causal loss scenarios.

---

## 🗺️ ODD Specification (Exercise 2.2)

```
╔══════════════════════════════════════════════════════════════╗
║              Operational Design Domain (ODD)                ║
╠══════════════╦══════════════════════╦═══════════════════════╣
║ Dimension    ║ Valid (IN ODD)        ║ Invalid (OUT OF ODD)  ║
╠══════════════╬══════════════════════╬═══════════════════════╣
║ Weather      ║ Clear, dry           ║ Fog, rain, snow       ║
║ Lighting     ║ Daytime, full sun    ║ Night, dusk, glare    ║
║ Camera       ║ Clean lens           ║ Dirty, occluded       ║
║ Scene type   ║ Urban road           ║ Motorway, tunnel      ║
║ Speed        ║ ≤ 50 km/h            ║ > 50 km/h             ║
╚══════════════╩══════════════════════╩═══════════════════════╝
```

---

## ⚠️ Losses & Hazards

**Losses (what must be prevented):**

| ID | Loss | Why Unacceptable |
|---|---|---|
| L-1 | Injury or death of pedestrian / road user | Irreversible harm |
| L-2 | Damage to ego vehicle or third-party property | Financial + legal |
| L-3 | Regulatory / legal violation | Liability |

**System Hazards:**

| ID | Hazard | Loss(es) | Likelihood | Severity |
|---|---|---|---|---|
| H-1 | Vehicle fails to brake for a pedestrian | L-1, L-2 | Medium | High |
| H-2 | Vehicle runs a red light into cross-traffic | L-1, L-2, L-3 | Low | High |
| H-3 | Unsafe overtake due to missed vehicle detection | L-1, L-2 | Medium | High |
| H-4 | System operates outside ODD undetected | L-1, L-2, L-3 | High | High |

---

## 🏗️ Control Structure (Exercise 2.5)

```
         ┌──────────────┐
         │ Human Operator│
         └──────┬───────┘
    Override ↓  ↑ Alerts
         ┌──────┴───────┐
         │   Planning    │◄── Distance Oracle
         │   Module      │◄── Vehicle Speed
         └──────┬───────┘
    Commands ↓  ↑ Perception Labels
         ┌──────┴───────┐         ┌──────────────────┐
         │   Actuators   │         │  Perception Layer │
         │(brake/steer)  │         │ ┌──────────────┐ │
         └──────────────┘         │ │Traffic Light │ │
                                  │ │  Model       │ │
              ┌─────────┐         │ ├──────────────┤ │
              │  Camera  │────────►│ │  Pedestrian  │ │
              └─────────┘         │ │  Model       │ │
                                  │ ├──────────────┤ │
                                  │ │  Vehicle     │ │
                                  │ │  Model       │ │
                                  │ └──────────────┘ │
                                  └──────────────────┘
```

---

## 🚨 Unsafe Control Actions (Exercise 2.6)

| ID | Controller | Control Action | UCA Type | Hazard | Scenario |
|---|---|---|---|---|---|
| UCA-1 | Planner | Issue brake command | Not provided | H-1 | Pedestrian model outputs 0 (missed) |
| UCA-2 | Planner | Maintain speed | Provided unsafely | H-1, H-4 | Input is OOD, perception unreliable |
| UCA-3 | Planner | Issue stop command | Not provided | H-2 | Traffic light missed in fog |
| UCA-4 | Human Op | Override autopilot | Wrong timing | H-1 | Operator overrides braking too late |

---

## 🛡️ Safety Constraints (Exercise 2.7)

| UCA | Constraint | Level | Verification |
|---|---|---|---|
| UCA-1 | Pedestrian recall ≥ minimum threshold | Model-level | Sheet 4 metrics |
| UCA-2 | OOD monitor must flag out-of-ODD inputs | Model-level | Sheet 9 AUROC |
| UCA-2 | If OOD flag raised → reduce speed to ≤ 15 km/h | System-level | Design review |
| UCA-3 | Traffic light recall ≥ threshold | Model-level | Sheet 4 metrics |
| UCA-4 | Alert human operator when confidence < θ | System-level | Design review |

---

*→ Next: [Exercise Sheet 3 — Baseline Training](README_Exercise3.md)*
