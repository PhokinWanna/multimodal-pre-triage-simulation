# M3 experiment protocol (fixed before any results)

**Status:** pre-registered on 2026-11-02, before the photos were collected, the reference labels were agreed, or any M3 run. Any later change is listed in the change log at the end, with its reason.

## 1. Question

How well does the full multimodal pipeline **M3** (patient text + one camera frame → posture rules + skin CNN → local LLM) assign the ESI level of scripted patient scenarios, compared with an agreed reference level? How often does it under-triage (choose a less urgent level than the reference)?

## 2. Configurations

| Name | Inputs to the LLM | Status |
| --- | --- | --- |
| M0 | text only | later |
| M1 | text + posture | later |
| M2 | text + skin CNN | later |
| **M3** | **text + posture + skin CNN** | **this experiment** |

M0–M2 will reuse the same scenarios and photos (`run_m3.py --no-posture/--no-skin`).

## 3. Cases

- **30 scenarios**, 6 per reference ESI level, 15 Thai and 15 English (`scenarios.csv`). Original texts written from ESI criteria (not copied from the ESI Handbook).
- **One test photo per scenario**, matched by category (`IMAGE_CHECKLIST.md`): 8 neither, 8 posture only, 7 flushing only. The **7 "both" scenarios use two photos**: posture from a body photo, skin result from a separate flushed-face photo (37 test photos in total).
- Photos come from openly licensed sources, never from the CNN's training datasets, and pass `check_duplicates.py`.
- The photo's true content (`true_posture`, `true_flushing`) is recorded by the researcher and checked by a second person.

## 4. Reference standard

1. Claude drafted a level for each scenario with its ESI decision point (sealed in `sealed/labels_claude_draft.csv`).
2. Phokin labels all 30 independently, without opening the sealed file (`labels_phokin.csv`).
3. `compare_labels.py` reports agreement (exact and weighted kappa) and lists disagreements.
4. Disagreements are decided by Ajarn Komsan (and, if possible, a nurse or nursing lecturer).
5. Independent check: Ajarn Komsan labels the 30 cases on a shuffled, renumbered sheet without levels (`blind_review_ajarn.csv`); agreement with the reference is reported (exact, weighted kappa), and M3 is also scored against his labels as a sensitivity analysis.

**Limitation:** the reference is criterion-based and agreed by the researchers. It has not been validated by triage nurses.

## 5. Pipeline settings (fixed)

| Setting | Value |
| --- | --- |
| LLM | `llama3.2:latest` via Ollama, the kiosk's final-verdict prompt unchanged |
| Option A (main) | temperature 0, 1 run per scenario, seed 0 |
| Option B (supporting) | temperature 0.8, 10 runs per scenario, seeds 1–10 |
| LLM timeout | 60 s. Latency is recorded and calls over the kiosk's 6 s limit are counted. |
| Camera | single frame, static MediaPipe, no 3-second hold |
| Posture rules | v2 (corrected). Original v1 run as appendix. |
| Skin | forehead + cheek 64×64 patches as in training, mean probability, SkinCNN-FL-α0.75 best checkpoint (epoch 33), converted to TFLite with a match test, threshold 0.549 (validation-selected, precision ≥ 0.95). Original pipeline (whole face, original model, 0.65) run as appendix. |
| Exit reason sent | `NORMAL_COMPLETE` |
| Tuning | posture/skin thresholds may be adjusted on the 14 development photos ONLY, then frozen before the run |

## 6. Outcomes

**Primary:** exact accuracy and linearly weighted kappa (option A).

**Safety:** under-triage rate; sensitivity and specificity for high acuity (reference ESI 1–2).

**Also reported:**
- within-one-level accuracy;
- quadratic kappa;
- over-triage rate;
- AUC for high acuity:
  - option A: score = 6 − predicted level;
  - option B: score = share of the 10 runs predicting ESI 1–2.
- option B majority-vote metrics and consistency;
- camera-part accuracy (posture, skin) and the first failing part of each wrong answer;
- latency and fallback counts.

All proportions and kappas come with 95% percentile bootstrap intervals (2,000 resamples over scenarios, seed 42).

## 7. Rules decided in advance

- An invalid or missing LLM level counts as **wrong** and is reported.
- A majority-vote tie goes to the **more urgent** level.
- A case where the camera finds no face gets skin = `UNKNOWN` and is still included.
- No scenario, photo or label is changed after the first M3 run. Any re-run is reported.

## 8. Known risks to report

- n = 30: confidence intervals will be wide.
- Single photos, not video; stock photos may be posed.
- In the 7 "both" scenarios, posture and skin come from two different people's photos; a real kiosk sees both in one frame.
- Text input skips speech recognition errors.
- **Skin CNN shortcut risk:** early probing showed the model reacts strongly to patch smoothness/blur and little to redness on synthetic patches. `skin_shortcut_check.py` measures this on real patches before the run; the result is reported whatever it shows.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-11-02 | Protocol created | — |
| 2026-11-03 | Blind review by Ajarn Komsan **not done**: he declined because triage labelling is outside his expertise. No clinical rater was available. | Reported as a limitation. |
| 2026-11-03 | Disclosure: Phokin assigned his labels **with the assistance of an AI language model** prompted to act as an experienced ED/OPD triage nurse (tool: Claude Opus 5.5), then reviewed and accepted them. Claude's sealed draft came from the same model family, so the 30/30 agreement is NOT independent inter-rater agreement and is not reported as such. The system under test uses a different model (Llama 3.2), so the evaluation is not circular, but the reference standard may still share LLM blind spots; results measure agreement with a criterion-based, AI-assisted reference, not clinical accuracy. | Required for an accurate method description. |
| 2026-11-03 | 6 round-B development photos added (researcher's own seated photos: chest ×2, abdomen ×2, head ×2). Not used for round A, whose rules were already frozen. | For round B (improvement). |
| 2026-11-03 | Reference labels: Phokin and Claude agreed on 30/30 (weighted kappa 1.000) → `labels_final.csv` (all "agreed"). Limitation found: scenario IDs were ordered by ESI level (S01–S06 = 1, …), which could reveal the design, so the two label sets were not fully independent. Added: Ajarn Komsan labels a shuffled, renumbered sheet (`blind_review_ajarn.csv`, key sealed). The reference standard stays `labels_final.csv`; Ajarn's blind labels are reported as independent inter-rater agreement and used as a secondary reference in a sensitivity analysis. Decided before any M3 run. | Honest correction of a design flaw. |
| 2026-11-03 | Posture v2 tuned on the 14 development photos, then **frozen**: chest radius 0.30 → 0.24; torso estimate uses head-to-shoulder height when hips are not visible; crossed arms also detected when both wrists cross the body midline. Dev result: fixed rules 11/14 correct (chest poses 3/4), original rules 10/14 (chest poses 1/4). Remaining dev errors (folded arms read as chest; hand on lower ribs read as abdomen; head-scratch read as headache) are ambiguous in 2D and were not tuned further, to avoid overfitting. | Allowed step 6 of this protocol; no test photo was analysed. |
| 2026-11-03 | Skin shortcut check on 200 Normal + 200 Flushing real training patches: slight blur turned 84% of Normal patches into "Flushing"; slight grain turned 68.5% of Flushing patches into "Normal"; changing redness (CIELAB a* ±12) changed ≤11.5%. On the dev photos the patch pipeline called 9/13 normal faces "Flushing". Training data inspection: Flushing images are mostly DermNet clinical close-ups, Normal images ordinary face photos. | Pre-planned check (section 8). Result reported as found. |
| 2026-11-02 | "Both" scenarios: posture from a `_body` photo + skin from a separate `_face` photo (was: one photo showing both) | Openly licensed photos showing a red face and a pain posture together could not be found. The camera produces posture and skin as separate results, so the LLM input is unchanged. Decided before any photo was analysed or any run. |
| 2026-11-14 | **Round B amendment** — written before the round-B test run (see section "Round B" below). | Ajarn's review of round A (2026-11-14). |

## Round B (amendment 2026-11-14, fixed before the round-B test run)

**Question.** Does the corrected system (round B) agree better with the reference on the **same 30 scenarios, photos and reference labels** as round A? How do posture:skin weights of 50:50, 60:40 and 40:60 change the result, and how much does each input (patient's words, posture, skin) actually move the answer?

**Scope.** Reference ESI 1 = outside the kiosk's scope (the patient needs staff at once). These 6 scenarios are scored as *out-of-scope detection*: answered ESI 1 (strict) and answered ESI 1–2 (urgent alert, lenient). **Primary outcomes use the 24 in-scope scenarios (reference ESI 2–5):** exact accuracy and linearly weighted kappa; safety: under-triage, ESI 2 found (answered 1–2); also over-triage and ESI 5 recall. All-30 figures are also reported for comparison with round A.

**System changes (all tuned on development data only; no test photo or test scenario was analysed while tuning):**

| Part | Round A | Round B | Development data used |
| --- | --- | --- | --- |
| Posture rules | v2 | **v3**: chest zone re-fitted, one hand = one zone, extrapolated hips when out of frame, temple counts as head | Roboflow "Medical Emergency Detection" v2 pool, one image per source photo, 9 sources that are M3 test photos excluded by byte match (and confirmed by a perceptual-hash check); 70 % tuning / 30 % hold-out. Hold-out chest sensitivity 67.0 % → 74.6 %, specificity 87.5 % → 87.1 %. The 20 dev photos: 16/20 for both v2 and v3. |
| Skin | patches, always answers | **patches + image-suitability check** (`patch_qc`): no answer when the face is narrower than 200 px, a patch is blurred (Laplacian variance < 30), too dark/bright; faces wider than the Normal training class (> 300 px) are answered only when the CNN agrees at the original size and reduced to training size | CNN training images only (`skin_quality_study.py`, parts A–D). Dev photos: round-A pipeline called 9/19 normal faces "Flushing", round B 0/4 answered. Cost: on wide Flushing training photos round B answers only about 26 % (it refuses rather than guesses). |
| Final-verdict prompt | v1 | chosen among v1 / v2 / v3 (ESI decision steps A–C, explicit ESI 5, worked examples) and "level field" vs "level implied by the step answers" | 15 development scenarios (`dev_scenarios.csv`, 3 per level, written for this purpose, none shares a text with the test set). Rule fixed in advance: highest linear kappa; within 0.01 → fewer under-triaged → fewer invalid outputs → simpler candidate. Recorded in `results/experiments/m3/roundB/prompt_selection.json`. |
| Camera fusion | posture and skin separate | **weighted visual score** = wP·posture + wS·skin (posture: chest 1.0, abdomen 0.7, head 0.6; skin = CNN probability; a part not assessed adds 0 and keeps its weight). Bands: HIGH ≥ 0.50, MODERATE ≥ 0.25. Safety floor: HIGH → at most ESI 2. | Weights 50:50 = main system (decided before the run); 60:40 and 40:60 as Ajarn asked. |
| Output | ESI | ESI + `scope` (ESI 1 → "out of kiosk scope, staff now") + a decision record (LLM answer, rule answer, floor applied) | — |

Unchanged: the 30 scenarios, the 37 photos, `labels_final.csv`, the LLM (`llama3.2:latest`), temperature 0 / seed 0 for the main run, the metrics code and bootstrap settings.

**Configurations** (`run_roundB.py`, each scenario once): W50 (main), W60, W40, IND (posture and skin as two separate findings, new prompt), the 8 on/off combinations of words / posture / skin (for the Shapley input-contribution analysis), and W50/W60/W40/IND through the rule path only (no LLM). Optional: 10 sampled runs of W50 (temperature 0.8, seeds 1–10).

**Analysis** (`analyze_roundB.py`): the tables above with 95 % bootstrap intervals; per-input Shapley contribution (average absolute change in ESI levels and share) for the LLM alone and for the whole system; sensitivity to the floor policy (none / HIGH only / HIGH and MODERATE); camera accuracy per round.

**Known limits added in round B.** The same 30 scenarios were seen in round A, so round B is not an unseen test: the improvements were made on development data, but the decision of *what* to improve came from round-A errors. The ESI 1 out-of-scope rule was introduced after round A (Ajarn's advice). The weights 60:40 / 40:60 are compared descriptively; with 24 in-scope scenarios small differences are not meaningful.

### Amendment 2 (2026-11-15, after the first round-B run)

Two things were found in the first round-B run (results kept in git history; the run folder is re-used, finished answers are read from the cache):

1. **Cut-off LLM answers (engineering bug).** 15 of 330 LLM answers (2 of 30 in the main system) were not valid JSON. Round A used a 256-token budget; the kiosk had since been changed to 200 tokens for speed, which can cut off a long Thai note. Fix, in the kiosk itself: when the JSON is broken, ask once more with twice the budget. Only those calls are repeated. The prompt, the scenarios and the selection are unchanged.
2. **The skin image check refused most test photos** (answered 8 of 30; found 2 of 14 truly flushed faces). The main system stays as registered. Because this leaves the posture : skin weighting with almost no skin input, an **exploratory** sensitivity analysis is added: the weighted configurations, and the configurations needed for the input-share analysis, are repeated with the round-A skin pipeline (patches without the image check, which answers for almost every photo but called 8 of 16 normal faces flushed in round A). Reported separately and labelled exploratory; it does not replace the main result.
