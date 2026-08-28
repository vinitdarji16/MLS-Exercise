# Exercise 9: Anomaly Detection (Out-of-Distribution Detection)

This file documents Exercise Sheet 9 in full: the OOD problem, MSP and feature-based
detection methods, and the anomaly-detection analysis for all three CARLA perception
classifiers.

*(Note on renumbering: this content was previously filed in this repository as
untitled/unnumbered "OOD Detection." The official sheet numbering places anomaly
detection at **Sheet 9**; calibration is **Sheet 7**, documented separately in
`README_Exercise7_Calibration.md`. The underlying measurements are unchanged; only the
exercise attribution and the missing conceptual/STPA sections have been added.)*

---

## Exercise 9.1: The OOD Problem

1. **Why can't a standard classifier be trusted to signal an OOD input?** A standard
   classifier is trained to minimize loss over its training distribution and has no
   mechanism that represents "I have never seen anything like this." Its softmax/sigmoid
   output is a relative comparison forced onto the classes it knows about — it will
   still emit a confident-looking number for an input from a completely different
   distribution, because nothing in its training objective ever taught it to output low
   confidence specifically *because* an input is unfamiliar, as opposed to because the
   input is genuinely ambiguous between its known classes.
2. **Why is silent failure worse than uncertain failure?** An uncertain failure (the
   system reports low confidence, or flags itself as unreliable) gives the downstream
   planner or human operator a signal to fall back to a conservative action. A silent,
   confidently-wrong failure gives no such signal — the system proceeds as if
   everything is normal while its perception is actually unreliable, removing the
   opportunity for any mitigation to engage before harm occurs. This is precisely the
   failure mode measured below: the pedestrian model's confidence moves in the *wrong*
   direction at night (Exercise 9.4), making it actively misleading rather than merely
   unhelpful.

## Exercise 9.2: Baseline OOD Detection

**Maximum Softmax Probability (MSP)** uses the model's own confidence as the OOD
signal — for a binary sigmoid output $p$, $s_{\text{MSP}}(\mathbf{x}) = \max(p, 1-p)$,
with lower confidence assumed to indicate an OOD input. **Main limitations:** it
assumes the model's confidence degrades gracefully outside the training distribution,
which is not guaranteed and is empirically false here for the pedestrian model at
night (Exercise 9.4/9.6); it conflates aleatoric uncertainty (genuine scene ambiguity,
Exercise 7.1) with epistemic/distributional uncertainty, since both lower $p$'s
extremity; and it requires no additional infrastructure but also provides no separate
signal beyond what the classifier was already producing, so any miscalibration in the
base classifier (Exercise 7) directly corrupts the OOD signal too.

## Exercise 9.3: Alternative Methods

**k-NN feature distance** (the advanced method used throughout this repository)
extracts penultimate-layer features for a reference set of in-distribution training
images, then scores a new input by its distance to its $k$-th nearest neighbor in that
feature space ($k=5$, Euclidean, throughout). **Improvement over MSP:** it operates in
representation space rather than relying on the classifier's final decision layer, so
it can flag an input as anomalous even when the classifier's output layer happens to
still produce a confident (but wrong) prediction — it is not coupled to the same
calibration failures that corrupt MSP. It does not require the classifier to have been
trained with any uncertainty-awareness at all, only a feature extractor.

---

## Practical: OOD Detection for the CARLA Model

## Exercise 9.4: Visualising the Distribution Shift

Five in-distribution, five fog, five night, and five Town-01 images are displayed
side-by-side per model (`show_images()` / grid-of-subplots cells in each notebook).
Qualitatively, fog images show a uniform desaturating haze, night images are globally
dark with strong artificial-light hotspots, and Town-01 images share the same
sunny/daytime photometric statistics as the training town but a visibly different
building layout, road geometry, and vegetation — i.e. Town-01 is a **structural** shift
while fog/night are **photometric** shifts, a distinction that matters for question 9.5.

**Mean confidence, in-distribution vs. each OOD condition:**

| Model | ID Test | Fog | Night | Town-01 |
| :--- | :--- | :--- | :--- | :--- |
| Pedestrian | 0.1650 | 0.0942 | **0.0151** | 0.1789 |
| Vehicle | 0.6604 | 0.4255 | 0.6643 | 0.5317 |
| Traffic light | 0.7766 | 0.4620 | **0.0025** | 0.7808 |

**Is the model more or less confident on these inputs?** It depends sharply on which
model and which condition — there is no single answer. Fog reduces confidence for all
three models. Night has the most extreme and inconsistent effect: pedestrian and
traffic-light confidence both collapse toward zero, while vehicle confidence is
essentially *unchanged* (0.6604 → 0.6643). Town-01 barely moves confidence for any
model, despite being a real distributional shift (different map) — consistent with
Exercise 9.4's photometric-vs-structural distinction: confidence-based signals are
sensitive to appearance shift but nearly blind to geometric/structural shift.

## Exercise 9.5: Is the Different Town Out-of-Distribution?

1. **Does the Exercise 2.2 ODD clearly decide this?** No. Exercise 2.2 defined the
   "operating conditions" scene dimension as "mapped urban roads and intersections"
   without naming a specific town, so a literal reading is ambiguous: Town-01 is a
   mapped urban road network, just not the one trained on.
2. **Revised wording, and which choice.** Revise to: *"Scene = the specific mapped
   road network used in training (Town-02) — any other town, mapped or not, is outside
   the ODD."* This resolves the ambiguity by making the ODD boundary about the
   *specific trained map*, not the general category "urban road." **Choice: Town-01 is
   outside the ODD**, justified because the confidence data above shows the model's
   behavior does not simply degrade gracefully in Town-01 the way a within-ODD
   variation would be expected to — it changes in the same unpredictable,
   model-specific way as the acknowledged OOD conditions (fog, night), and the
   AUROC data below confirms actual detection difficulty is real even where confidence
   alone doesn't show it.
3. **Implication.** Under this revised wording, Town-01 images are inputs an OOD
   monitor **should flag**, not inputs the system is expected to handle correctly on
   its own. This has a direct, uncomfortable consequence given Exercise 9.4/9.6/9.7's
   results: the monitors evaluated below are specifically *worst* at detecting the
   Town-01 shift (lowest AUROC gap of the three conditions for two of three models),
   meaning the input category this ODD revision assigns to "must be flagged" is the
   one the available detectors are least equipped to flag.

## Exercise 9.6–9.7: Evaluating MSP and Feature-Based (k-NN) Detection

**Combined AUROC (all OOD conditions pooled, 0 = chance, 1 = perfect):**

| Model | MSP AUROC | $k$-NN AUROC | Threshold | Met? |
| :--- | :--- | :--- | :--- | :--- |
| Pedestrian | 0.5096 | 0.5795 | ≥ 0.90 | ✗ **Not met** |
| Traffic light | 0.6855 | 0.8404 | ≥ 0.90 | ✗ Not met |
| Vehicle | 0.6172 | 0.5986 | ≥ 0.90 | ✗ Not met |

**Per-condition breakdown (MSP / k-NN):**

| Model | Fog | Night | Town-01 |
| :--- | :--- | :--- | :--- |
| Pedestrian | 0.6882 / 0.3433 | 0.2467 / 0.8850 | 0.5941 / 0.5102 |
| Traffic light | 0.9422 / 0.7653 | 0.3402 / 1.0000 | 0.7740 / 0.7560 |
| Vehicle | 0.3561 / 0.5748 | 0.3860 / 0.7261 | 0.4063 / 0.4950 |

**For which OOD scenario is the k-NN vs. MSP gap largest?** **Night**, by a wide
margin, for both the pedestrian model ($k$-NN $-$ MSP $= +0.64$) and the traffic-light
model ($+0.66$) — exactly the condition where each model's raw confidence was shown in
Exercise 9.4 to move in the unsafe direction. The vehicle model's largest gap is also
at night ($+0.34$), though smaller in absolute terms. Town-01 (the structural shift)
shows the *smallest* gap for all three models, and is sometimes negative (k-NN worse
than MSP, e.g. pedestrian fog: $-0.34$) — feature-space distance does not reliably
outperform confidence on the structural-shift condition the way it does on the
photometric ones.

### Key Insights

1. **No detector reaches the reliability threshold, for any classifier.** OOD
   violation cannot currently be assumed to be caught — this is a genuine open gap,
   not a near-miss.
2. **The pedestrian model's MSP confidence moves in the *wrong* direction at night**
   (AUROC 0.2467 — worse than random, in the unsafe direction): the model is *more*
   confident, not less, on out-of-distribution night frames. The traffic-light model
   shows the same qualitative failure at night (AUROC 0.3402). A naive
   confidence-based monitor would be actively misleading in both cases, not just
   unhelpful.
3. **$k$-NN outperforms MSP on appearance shifts (fog, night) but both struggle on
   geometric shift (Town-01)** — a different map with the same weather barely moves
   either detector's score, even though the underlying models' actual performance does
   degrade there (Exercise 4/8). This suggests neither monitor generalizes to
   *structural* distribution shift the way it does to *photometric* shift — directly
   relevant to the Exercise 9.5 conclusion that Town-01 is exactly the condition an OOD
   monitor is asked to catch.
4. **No single detector choice is safe across all three classifiers.** $k$-NN is
   near-perfect for traffic-light-at-night (1.0000) but mediocre for vehicle-at-night
   (0.7261) — a deployed system would need per-classifier monitor tuning, not one
   fixed recipe.

## Exercise 9.8: Extending the Safety Analysis for OOD

1. **Hazard.** H-5 in Exercise 2.4 ("the system remains confident on an input outside
   the training/ODD distribution") already captures this — its likelihood assessment
   ("High — no evaluated OOD monitor reaches reliable separation") is directly
   confirmed by the AUROC table above, none of which clears the 0.90 bar.
2. **Unsafe control action.** UCA-1 in Exercise 2.6 already covers the generic case;
   the specific, evidenced instantiation to add: "the planner continues at speed while
   the camera input is out-of-ODD (night) and the pedestrian/traffic-light perception
   output is untrustworthy, because MSP confidence at night is *higher*, not lower,
   than in-distribution confidence" — linked to H-1, H-3, H-5.
3. **Safety constraints.**
   - *Model-level:* an OOD monitor (the better of MSP/k-NN per classifier) shall
     achieve AUROC ≥ 0.90 separating in-distribution from each named OOD condition,
     evaluated per classifier and per condition rather than pooled (since pooled
     AUROC, as shown above, can mask a specific condition's monitor being actively
     unsafe).
   - *System-level:* when the OOD monitor score crosses its calibrated threshold, or
     when the input is known to originate outside the trained map/town, the system
     shall fall back to a conservative action rather than acting on perception output —
     independent of what the perception model's own confidence reports, since that
     confidence has been shown to be untrustworthy exactly when it matters most (night).
4. **Residual risk.** Even a perfect OOD detector only tells the system *that* an input
   is unfamiliar — it does not by itself make the system safe; a conservative fallback
   action must still exist and be correctly triggered (a system-level, not
   model-level, requirement — see Exercise 2.7's unresolved fallback constraint). The
   Town-01 finding (Exercise 9.5/9.6) additionally shows that "detecting OOD" and
   "detecting the specific OOD conditions a monitor was tuned for" are not the same
   thing — a monitor validated only on fog/night is not evidence it will catch an
   unanticipated structural shift, and the k-NN detector's mediocre Town-01
   performance (0.50–0.76) even after being fit on this exact test suite is a caution
   against assuming a monitor will generalize to conditions beyond what it was
   evaluated on.

---

*Notebooks: `MLS_exercise3_pedestrian_1.ipynb` (pedestrian OOD, cells 34–58),
`MLS_exercises_vehicle.ipynb` (vehicle OOD, cells 34–51), `MLS_exercises3_final_1.ipynb`
(traffic light OOD, cells 33–55) · shared code: `get_confidence()`, `get_msp_scores()`,
`extract_features()`, `roc_auc_score` (scikit-learn), `NearestNeighbors` (scikit-learn).*
