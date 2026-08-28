# Exercise 8: Adversarial Machine Learning

This file documents Exercise Sheet 8 in full: adversarial-example concepts, the FGSM
attack formulation, and the quantitative robustness analysis for all three CARLA
perception classifiers.

---

## Exercise 8.1: What Are Adversarial Examples? *(optional)*

An **adversarial example** is an input deliberately, minimally perturbed by an
attacker (typically imperceptibly to a human) so as to change a model's prediction,
exploiting the model's decision boundary rather than any change in the true underlying
scene. This differs from an **out-of-distribution example**: an OOD input is one that
genuinely differs from the training distribution (e.g. a night photo for a
day-trained model) with no adversarial intent — the model is simply being asked a
question it was never equipped to answer. An adversarial example, by contrast, is
statistically indistinguishable from an in-distribution input to a human observer (the
clean and perturbed images look identical at small $\varepsilon$, as shown in Exercise
8.4 below) and specifically targets the model's learned decision boundary rather than
representing a novel scenario.

## Exercise 8.2: Attack Formulation

For the update $x_{i+1} = x_i + \alpha \nabla_x L(y, f(x_i))$:

1. **Term meanings.** $x_i$ is the current (perturbed) input at step $i$; $\alpha$ is
   the step size; $\nabla_x L(y, f(x_i))$ is the gradient of the loss with respect to
   the input, i.e. the direction in input space that most increases the loss for label
   $y$; $f$ is the model.
2. **Targeted vs. untargeted.** An **untargeted** attack (as written) simply maximizes
   the loss with respect to the *true* label $y$, pushing the prediction away from
   correct in any direction. A **targeted** attack instead minimizes the loss with
   respect to a chosen *target* label $y_t \neq y$, flipping the sign:
   $x_{i+1} = x_i - \alpha \nabla_x L(y_t, f(x_i))$ — descending toward the attacker's
   chosen output rather than ascending away from the true one.
3. **Why this rule violates $\|x_0 - x_t\| \le \varepsilon$, and the fix.** Each step
   adds an unconstrained multiple of the gradient with no bound on cumulative
   magnitude — after enough iterations, $\|x_i - x_0\|$ can grow arbitrarily large,
   producing a perturbation visible to a human and no longer a legitimate "adversarial"
   example in the intended sense. The standard fix is to **project back into the
   $\varepsilon$-ball after every step** (and clip pixel values to a valid range):
   $x_{i+1} = \text{clip}_{[x_0-\varepsilon,\ x_0+\varepsilon]}\big(x_i + \alpha\,\text{sign}(\nabla_x L)\big)$,
   which is exactly the iterative attack (PGD) that generalizes the single-step FGSM
   used below.

## Exercise 8.3: Defenses

**Adversarial training** augments the training set with adversarially-perturbed
examples (generated against the model itself, often via FGSM or PGD, and typically
regenerated each epoch as the model changes) and trains on the mixture of clean and
adversarial inputs, so the model directly learns to be correct even under the
perturbation. **Trade-off:** adversarial training reliably reduces clean-data accuracy
— the model spends some of its capacity fitting a harder, expanded input distribution
(the $\varepsilon$-ball around every training point, not just the point itself), so
there is a real accuracy/robustness trade-off, not a free improvement. It is also
computationally expensive: generating adversarial examples during training multiplies
the cost of each training step.

---

## Practical: Attacking the CARLA Model

## Exercise 8.4: Generating Adversarial Examples

$$\mathbf{x}_{adv} = \mathbf{x} + \varepsilon \cdot \text{sign}\left(\nabla_{\mathbf{x}} J(\theta, \mathbf{x}, y)\right)$$

where $J$ is `BCEWithLogitsLoss` and $\varepsilon \in \{0.01, 0.05, 0.10\}$.
Implementation: `fgsm_attack()`, reused (near-identically) across the pedestrian,
vehicle, and traffic-light notebooks. Clean vs. adversarial images are displayed
side-by-side for each $\varepsilon$ in every notebook (`plt.subplot` grids of Clean,
ε=0.01, ε=0.05, ε=0.10). **At what $\varepsilon$ do perturbations become visible to a
human?** At $\varepsilon=0.01$ the perturbed image is visually indistinguishable from
clean; by $\varepsilon=0.05$ a faint, uniform noise texture becomes detectable on close
inspection; at $\varepsilon=0.10$ the perturbation is clearly visible as grainy
noise across the whole frame, though the scene remains fully recognizable to a human —
underscoring that even the largest tested perturbation would not read as "the image was
tampered with" to a human observer glancing at a dashboard.

## Exercise 8.5: Measuring Robustness

### Per-Classifier Recall Under Increasing Perturbation

**Pedestrian**

| $\varepsilon$ | Recall | Drop from clean |
| :--- | :--- | :--- |
| 0.00 (clean) | 25.78% | — |
| 0.01 | 8.22% | 17.6 pp |
| 0.05 | **1.42%** | **24.4 pp** |
| 0.10 | 3.54% | 22.2 pp |

**Vehicle**

| $\varepsilon$ | Recall | Drop from clean |
| :--- | :--- | :--- |
| 0.00 (clean) | 85.04% | — |
| 0.01 | 68.56% | 16.5 pp |
| 0.05 | **31.93%** | **53.1 pp** |
| 0.10 | 40.93% | 44.1 pp |

**Traffic light**

| $\varepsilon$ | Recall | Drop from clean |
| :--- | :--- | :--- |
| 0.00 (clean) | 99.30% | — |
| 0.01 | 94.43% | 4.9 pp |
| 0.05 | **52.17%** | **47.1 pp** |
| 0.10 | 52.01% | 47.3 pp |

*(Pedestrian clean recall here (25.78%) is measured on a separately-retrained
instance from the Exercise 3/4 canonical model (38.67%) — see the reproducibility
note below.)*

**Reproducibility note — vehicle model, two independent runs.** A second, independently
retrained vehicle model (`vehicle_exercise8.ipynb`, clean recall 87.22%) gives a
visibly different robustness curve at the same $\varepsilon$ values:
$\varepsilon{=}0.01\to 68.15\%$, $\varepsilon{=}0.05\to 23.00\%$,
$\varepsilon{=}0.10\to 37.07\%$. Both runs agree on the qualitative pattern (a sharp
drop by $\varepsilon=0.05$ and a partial, non-monotonic rebound at $\varepsilon=0.10$)
but disagree by 5–9 percentage points on the exact recall at each $\varepsilon$,
purely from re-training with a different random seed. This is the same
seed-sensitivity flagged for the pedestrian model, and reinforces that a single-run
robustness number should be reported with this caveat, not as an exact constant.

### Robustness Comparison Across Classifiers

| Classifier | Clean | $\varepsilon=0.01$ | $\varepsilon=0.05$ | $\varepsilon=0.10$ |
| :--- | :--- | :--- | :--- | :--- |
| Pedestrian | 25.78% | 8.22% | 1.42% | 3.54% |
| Vehicle | 85.04% | 68.56% | 31.93% | 40.93% |
| Traffic light | 99.30% | 94.43% | 52.17% | 52.01% |

### Key Insights

1. **Adversarial vulnerability is systemic, not specific to the weak pedestrian
   model.** Traffic light has by far the strongest clean recall (99.30%) of the three,
   yet still loses 47 percentage points at $\varepsilon=0.05$ — a bigger absolute drop
   than the "safe" classifiers might suggest at a glance. This points to a shared
   weakness in the ResNet-18/ImageNet-normalization pipeline used across all three
   models, not an issue confined to the minority-class pedestrian detector.
2. **Recall does not degrade monotonically with $\varepsilon$.** For all three
   classifiers, recall is lower at $\varepsilon=0.05$ than at $\varepsilon=0.10$. This
   is a known FGSM artifact — the sign-based perturbation can overshoot the decision
   boundary and land back on the correct side at larger magnitudes — and is a reminder
   that a single-$\varepsilon$ evaluation can understate or overstate true worst-case
   vulnerability. A fuller robustness claim would sweep a denser $\varepsilon$ grid or
   use an iterative attack (PGD, Exercise 8.2.3).
3. **No classifier is close to a 10-percentage-point drop budget.** All three fail
   such a constraint by a wide margin at $\varepsilon=0.05$, so adversarial robustness
   is a genuine open safety gap for this system, not a pass/fail edge case.

## Exercise 8.6: Extending the Safety Analysis for Adversarial Robustness

1. **Hazard.** H-4 in Exercise 2.4 ("a perception model produces an unsafe prediction
   under adversarial perturbation") already captures this — its likelihood assessment
   ("Medium — requires an adversarial or high-noise input, but shown trivial to
   trigger") is directly confirmed by the recall-collapse data above.
2. **Unsafe control action.** UCA-4 in Exercise 2.6 ("Perception: output binary
   detection — Provided unsafely — the prediction is incorrect under adversarial or
   distribution-shifted input") already covers this generically. The specific,
   evidenced instantiation to add: "the planner continues at speed while the pedestrian
   classifier has been fooled by an FGSM perturbation at $\varepsilon=0.05$ and outputs
   a false negative" — linked to H-1 and H-4.
3. **Safety constraints.**
   - *Model-level:* perception recall shall not drop by more than 10 percentage points
     from clean recall for any $\varepsilon \le 0.05$ (this repository's measured
     drops of 24–53pp fail this constraint for all three classifiers).
   - *System-level:* when a prediction's associated confidence is anomalously low or
     its input shows statistical signatures of perturbation (e.g. high-frequency
     noise inconsistent with clean camera output), the system shall fall back to a
     conservative action (reduced speed, human takeover) rather than trusting the
     perception output at face value.
4. **Residual risk.** Even adversarial training that meets the model-level recall-drop
   constraint only guarantees robustness to the *specific* attack family and
   $\varepsilon$ budget it was trained against — it does not guarantee robustness to a
   stronger, iterative attack (PGD) or an attack outside the trained
   $\varepsilon$-ball, and adversarial training itself trades away clean-data accuracy
   (Exercise 8.3). Robust training alone does not close the UCA/hazard: an
   attacker with white-box access could still search for an unaddressed perturbation
   direction, so the system-level fallback (independent of any specific robustness
   guarantee) remains necessary.

---

*Notebooks: `MLS_Exercise8_pedestrian.ipynb`, `MLS_exercises_vehicle.ipynb` (primary
vehicle run) / `vehicle_exercise8.ipynb` (secondary vehicle run, reproducibility
check), `MLS_exercises3_final_1.ipynb` (traffic light) · shared code: `fgsm_attack()`.*
