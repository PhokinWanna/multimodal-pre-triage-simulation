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
