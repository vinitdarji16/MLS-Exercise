# Exercise 7: Uncertainty Quantification

This file documents Exercise Sheet 7 in full: epistemic vs. aleatoric uncertainty,
calibration/ECE, cost-optimal decision thresholds, and the calibration analysis for
all three CARLA perception classifiers.

*(Note on renumbering: this content was previously filed in this repository under the
name "Exercise 9 — Calibration." The official sheet numbering places calibration at
**Sheet 7** and out-of-distribution/anomaly detection at **Sheet 9**
— see `README_Exercise9_OOD.md`. The underlying measurements are unchanged from the
earlier version; only the exercise attribution and the missing conceptual/cost-analysis
sections have been corrected and added here.)*

---

## Exercise 7.1: Epistemic vs. Aleatoric Uncertainty *(optional)*

**Epistemic uncertainty** is uncertainty due to the model's limited knowledge — it
shrinks as more relevant training data is seen, and reflects "the model hasn't learned
enough to know." **Aleatoric uncertainty** is uncertainty inherent to the data itself
— irreducible noise or ambiguity present even for a perfect model (e.g. a genuinely
ambiguous, partially-occluded scene).

1. **More relevant to OOD inputs (e.g. night images for a day-trained model):**
   **epistemic.** A night image lies outside what the model has ever seen, so its
   uncertainty there reflects a knowledge gap that more (relevant) training data could
   close — not an inherent ambiguity in the scene itself.
2. **Dominant type on a correctly-classified, in-distribution image:** **aleatoric.**
   The model has seen plenty of similar examples (low epistemic uncertainty by
   definition, since it is in-distribution and correct), so whatever residual
   uncertainty remains is attributable to genuine scene ambiguity (e.g. partial
   occlusion, borderline pedestrian visibility) rather than a knowledge gap.

## Exercise 7.2: Calibration and ECE

A classifier is **well-calibrated** if, among all predictions it assigns a confidence
of $p$, the empirical fraction that are actually correct is also $p$ — e.g. among all
predictions made with 80% confidence, roughly 80% should be correct. **Expected
Calibration Error (ECE)** partitions predictions into $M$ equal-width confidence bins
and measures the weighted gap between confidence and accuracy in each:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where $B_m$ is the set of predictions falling in bin $m$, $\text{acc}(B_m)$ is that
bin's empirical accuracy, and $\text{conf}(B_m)$ is its mean predicted probability.
$M=10$ bins are used throughout this repository.

## Exercise 7.3: Cost-Optimal Downstream Decisions

For $p = p(\text{ped}\mid x)$, brake cost $C_{FP}=1$ (unnecessary brake when no
pedestrian) and continue cost $C_{FN}=100$ (missed pedestrian):

1. **Expected loss of each action:**
   $$E[L \mid \text{brake}] = (1-p)\cdot C_{FP}, \qquad E[L \mid \text{continue}] = p\cdot C_{FN}$$
2. **Threshold $\tau^*$ where both actions are equal:**
   $$(1-\tau^*)\,C_{FP} = \tau^*\,C_{FN} \;\;\Longrightarrow\;\; \tau^* = \frac{C_{FP}}{C_{FP}+C_{FN}}$$
   For $p > \tau^*$, brake; for $p < \tau^*$, continue.
3. **Numeric value:** $\tau^* = \dfrac{1}{1+100} = \dfrac{1}{101} \approx 0.0099$ —
   roughly a 1% predicted probability of a pedestrian is already enough to justify
   braking, in stark contrast to the naive $\tau = 0.5$ rule. This reflects the
   200-to-1 asymmetry between the cost of an unnecessary brake and the cost of a missed
   pedestrian.
4. **Why $\tau^*$ only yields the optimal decision when $p$ is well-calibrated.** The
   derivation above treats $p$ as *the true probability* that a pedestrian is present.
   If the model is overconfident (reports $p$ higher than the true frequency of being
   correct), the system will brake less often than the cost structure actually
   warrants — silently eroding the safety margin the threshold was designed to
   provide. If the model is underconfident, it will brake needlessly often. Either way,
   applying $\tau^*$ to an uncalibrated $p$ does not achieve the intended cost trade-off
   — it only achieves it in expectation *if* $p$ truly reflects empirical correctness
   frequency, which is exactly what ECE measures and what Exercise 7.4–7.5 test.

---

## Practical: Calibrating the CARLA Model

## Exercise 7.4–7.5: Measuring Calibration and Temperature Scaling

**Temperature scaling** rescales the pre-sigmoid logit $z$ by a single learned scalar
$T$ before the sigmoid: $\hat{p} = \sigma(z/T)$.

| Model | ECE (raw) | Best $T$ | ECE (scaled) | Threshold | Met? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pedestrian | 0.7665 | 2.0 | 0.7142 | < 0.05 | ✗ **Not met** |
| Traffic light | 0.2616 | 1.5 | 0.2495 | < 0.05 | ✗ Not met |
| Vehicle | 0.2177 | 2.0 | 0.1947 | < 0.05 | ✗ Not met |

**Are the models over- or underconfident? Consistent across all three?** All three are
**overconfident** (raw ECE far exceeds accuracy-confidence agreement, and every model's
best-fit $T$ is greater than 1, i.e. scaling *down* the logits — flattening
confidence toward 0.5 — reduces ECE). This direction is consistent across all three
models, but the *magnitude* is wildly inconsistent: the pedestrian model's ECE (0.77
raw) is roughly 3× the other two models', meaning its confidence values have almost no
relationship to correctness, not just a mild miscalibration.

**Methodology note — an inconsistency worth flagging.** Only the traffic-light model
was calibrated exactly as Exercise 7.5 specifies: $T$ optimized by a proper grid
search over $T \in \{0.5, 0.6, \dots, 3.0\}$ (step 0.1) minimizing **validation-set
NLL**, which is how it arrives at $T=1.5$ (a value not in $\{0.5,1.0,2.0\}$, confirming
a genuine search was run). The pedestrian and vehicle models' best $T$ was instead
selected as the lowest-ECE value among only the coarse three-point grid
$\{0.5,1.0,2.0\}$ carried over from Exercise 5.4 — a narrower search that is not
guaranteed to find the true NLL-minimizing $T$. **Practical consequence:** the
pedestrian and vehicle ECE-after-scaling figures above should be treated as upper
bounds on what proper NLL-based temperature scaling could achieve, not as the final
word — re-running the full $T \in \{0.5, \dots, 3.0\}$ NLL grid search (already
implemented and validated on the traffic-light model) on the other two models is a
straightforward, low-cost follow-up.

## Exercise 7.6: Cost-Optimal Decision in Practice

Using $C_{FN}=100$, $C_{FP}=1$, $\tau^*\approx 0.0099$ from Exercise 7.3, applied to
the pedestrian classifier:

$$L = C_{FN}\cdot \#FN + C_{FP}\cdot \#FP$$

| | $\tau = 0.5$ | $\tau = \tau^*\approx 0.0099$ |
| :--- | :--- | :--- |
| **Uncalibrated** ($T=1$) | $\#FN=433,\ \#FP=196 \Rightarrow L = 100(433)+1(196) = \mathbf{43{,}496}$ | *not computed in the provided notebooks — see below* |
| **Calibrated** ($T=2.0$) | *not separately re-thresholded in the provided notebooks* | *not computed in the provided notebooks — see below* |

The $\tau=0.5$/uncalibrated cell is fully determined from Exercise 4.7's confusion
matrix (FN=433, FP=196 on N=3,600). **The other three cells require re-thresholding
the model's per-image probability array at $\tau^*\approx 0.0099$ — a sweep that was
not run in any of the provided notebooks** (they only ever evaluate at the default 0.5
threshold, whether raw or temperature-scaled). This is flagged explicitly rather than
estimated, since fabricating FN/FP counts at an untested threshold would misrepresent
what has actually been measured. To complete this table: reuse the `all_probs` /
`scaled_probs` arrays already computed in the pedestrian notebook's calibration cells,
threshold each at 0.0099 instead of 0.5, and recompute the confusion counts — no new
training or inference is required, only a different threshold applied to results
already on disk.

**Expected direction (qualitative, from the model's probability distribution shape):**
lowering the threshold from 0.5 to ~0.01 will convert most current false negatives into
true positives (since $C_{FN}\gg C_{FP}$ rewards this heavily) at the cost of a sharp
rise in false positives — this is very likely still the lowest-total-loss cell given
the 100:1 cost ratio, but the exact magnitude is not asserted here without the
re-thresholded counts.

## Exercise 7.7: Tracing Overconfidence Through the Safety Analysis

1. **Causal scenario (added to Exercise 2.8's causal loss scenarios):** the pedestrian
   classifier produces a false negative yet reports high confidence, so the planner
   registers "no pedestrian, reliable" and no low-confidence fallback is triggered.
   This is directly evidenced by the pedestrian model's raw ECE of 0.7665 — the largest
   miscalibration of any classifier in this system — meaning its confidence values are
   the least trustworthy signal in the entire perception stack precisely for the class
   where a false negative is most costly.
2. **Safety constraints derived from this scenario:**
   - *Model-level:* SC-3 (Exercise 4.6) — confidence estimates shall achieve ECE < 0.05
     after temperature scaling, evaluated per-classifier.
   - *System-level:* the planner shall use the cost-optimal threshold $\tau^*\approx
     0.0099$ (Exercise 7.3) rather than the naive $\tau=0.5$ for the pedestrian
     detector's brake decision, and shall trigger the reduce-speed fallback whenever
     confidence falls outside a range that has been empirically validated (via the ECE
     reliability diagram, not assumed) to track true correctness.
3. **Verification.** The evidence is produced directly on this sheet: raw and
   post-scaling ECE for all three models (Exercise 7.4–7.5). **The model-level
   constraint (ECE < 0.05) is not met by any classifier, even after temperature
   scaling** — the pedestrian model in particular remains at 0.7142, more than an
   order of magnitude above threshold. The calibrated model does **not** meet the
   threshold it was set.
4. **Residual risk.** Even a perfectly calibrated model only guarantees that confidence
   values *statistically* track correctness in aggregate — it does not guarantee any
   individual prediction is correct, and it says nothing about calibration holding up
   under distribution shift (Exercise 9 shows the pedestrian model's confidence
   behavior at night is actively misleading, not just uncalibrated in-distribution).
   Calibration alone does not close this loss scenario: the system-level fallback
   (reduced speed / human takeover when confidence is low, and use of $\tau^*$ instead
   of 0.5) remains required regardless of how well-calibrated the model becomes,
   because calibration is a statistical property of the confidence signal, not a
   correctness guarantee for any single frame.

---

*Notebooks: `trafficlight_exercise9.ipynb` (full NLL-based $T$ grid search, cells
17–19), `MLS_Exercise8_pedestrian.ipynb` (pedestrian ECE/temperature, cells 30–37),
`vehicle_exercise8.ipynb` (vehicle ECE/temperature, cells 32–40) · shared code:
`compute_ece()`, reused across all three notebooks with an identical implementation.*
