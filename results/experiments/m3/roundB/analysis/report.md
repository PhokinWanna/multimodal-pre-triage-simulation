# M3 round B results

Run: `results/experiments/m3/roundB/run_main` · prompt **v1** · llama3.2:latest · posture rules v3 · skin pipeline patch_qc · n = 30 scenarios (24 in scope, 6 reference ESI 1).

Intervals are 95 % percentile bootstrap intervals over scenarios (2,000 resamples).


## 1. In-scope scenarios (reference ESI 2–5, n = 24)

| | Round A, original pipeline | Round A, fixed pipeline | Round B, LLM answer before the safety floor | Round B, independent camera parts | Round B, MAIN (weighted 50/50) |
| --- | --- | --- | --- | --- | --- |
| Exact accuracy | 33% (15%–54%) | 50% (30%–71%) | 54% (35%–74%) | 62% (42%–81%) | 62% (42%–81%) |
| Within one level | 88% (73%–100%) | 92% (78%–100%) | 92% (78%–100%) | 88% (72%–100%) | 88% (72%–100%) |
| Weighted kappa (linear) | 0.22 (0.06–0.40) | 0.47 (0.22–0.69) | 0.50 (0.26–0.71) | 0.56 (0.28–0.78) | 0.56 (0.28–0.78) |
| Under-triage | 46% (25%–65%) | 25% (8%–43%) | 17% (4%–32%) | 8% (0%–22%) | 8% (0%–22%) |
| Over-triage | 21% (5%–38%) | 25% (8%–43%) | 29% (12%–48%) | 29% (12%–48%) | 29% (12%–48%) |
| ESI 2 found (answered 1–2) | 0% (0%–0%) | 50% (0%–100%) | 33% (0%–75%) | 67% (22%–100%) | 67% (22%–100%) |
| ESI 3–5 not over-called (answered 3–5) | 100% (100%–100%) | 94% (81%–100%) | 94% (81%–100%) | 94% (81%–100%) | 94% (81%–100%) |
| ESI 5 answered as 5 | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) |

## 2. Out-of-scope scenarios (reference ESI 1, n = 6)

| | Round A, original pipeline | Round A, fixed pipeline | Round B, LLM answer before the safety floor | Round B, independent camera parts | Round B, MAIN (weighted 50/50) |
| --- | --- | --- | --- | --- | --- |
| Answered ESI 1 (sent to staff at once) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) |
| Answered ESI 1 or 2 (urgent alert) | 50% (0%–100%) | 83% (50%–100%) | 67% (25%–100%) | 67% (25%–100%) | 67% (25%–100%) |

## 3. All scenarios (n = 30), for comparison with round A

| | Round A, original pipeline | Round A, fixed pipeline | Round B, LLM answer before the safety floor | Round B, independent camera parts | Round B, MAIN (weighted 50/50) |
| --- | --- | --- | --- | --- | --- |
| Exact accuracy | 27% (13%–43%) | 40% (23%–60%) | 43% (27%–60%) | 50% (33%–67%) | 50% (33%–67%) |
| Within one level | 80% (63%–93%) | 90% (77%–100%) | 87% (73%–97%) | 83% (70%–97%) | 83% (70%–97%) |
| Weighted kappa (linear) | 0.32 (0.18–0.46) | 0.49 (0.32–0.65) | 0.48 (0.30–0.63) | 0.50 (0.30–0.68) | 0.50 (0.30–0.68) |
| Under-triage | 57% (40%–73%) | 40% (23%–57%) | 33% (17%–50%) | 27% (13%–43%) | 27% (13%–43%) |
| Over-triage | 17% (3%–30%) | 20% (7%–37%) | 23% (10%–40%) | 23% (10%–40%) | 23% (10%–40%) |
| High acuity (ESI 1–2) sensitivity | 25% (0%–50%) | 67% (38%–92%) | 50% (21%–80%) | 67% (38%–92%) | 67% (38%–92%) |
| High acuity (ESI 1–2) specificity | 100% (100%–100%) | 94% (81%–100%) | 94% (81%–100%) | 94% (81%–100%) | 94% (81%–100%) |

## 4. Posture : skin weights — LLM + safety floor

| | Independent (no weights) | Posture 50 : Skin 50 | Posture 60 : Skin 40 | Posture 40 : Skin 60 |
| --- | --- | --- | --- | --- |
| Exact accuracy (in scope) | 62% (42%–81%) | 62% (42%–81%) | 62% (42%–81%) | 62% (42%–81%) |
| Within one level (in scope) | 88% (72%–100%) | 88% (72%–100%) | 88% (72%–100%) | 88% (72%–100%) |
| Weighted kappa (linear) (in scope) | 0.56 (0.28–0.78) | 0.56 (0.28–0.78) | 0.56 (0.28–0.78) | 0.56 (0.28–0.78) |
| Under-triage (in scope) | 8% (0%–22%) | 8% (0%–22%) | 8% (0%–22%) | 8% (0%–22%) |
| Over-triage (in scope) | 29% (12%–48%) | 29% (12%–48%) | 29% (12%–48%) | 29% (12%–48%) |
| ESI 5 answered as 5 | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) |
| Reference ESI 1 answered 1–2 | 67% (25%–100%) | 67% (25%–100%) | 67% (25%–100%) | 67% (25%–100%) |
| Exact accuracy (all 30) | 50% (33%–67%) | 50% (33%–67%) | 50% (33%–67%) | 50% (33%–67%) |
| Under-triage (all 30) | 27% (13%–43%) | 27% (13%–43%) | 27% (13%–43%) | 27% (13%–43%) |
| Over-triage (all 30) | 23% (10%–40%) | 23% (10%–40%) | 23% (10%–40%) | 23% (10%–40%) |

## 5. Posture : skin weights — rule-based path only (no LLM)

| | Independent (no weights) — rules only | Posture 50 : Skin 50 — rules only | Posture 60 : Skin 40 — rules only | Posture 40 : Skin 60 — rules only |
| --- | --- | --- | --- | --- |
| Exact accuracy (in scope) | 38% (18%–58%) | 58% (38%–77%) | 62% (42%–81%) | 50% (29%–69%) |
| Within one level (in scope) | 71% (50%–88%) | 88% (72%–100%) | 88% (72%–100%) | 83% (67%–96%) |
| Weighted kappa (linear) (in scope) | 0.13 (-0.06–0.34) | 0.52 (0.22–0.74) | 0.56 (0.27–0.79) | 0.38 (0.10–0.62) |
| Under-triage (in scope) | 17% (4%–33%) | 21% (6%–39%) | 17% (4%–33%) | 29% (12%–50%) |
| Over-triage (in scope) | 46% (27%–67%) | 21% (5%–38%) | 21% (5%–38%) | 21% (5%–38%) |
| ESI 5 answered as 5 | 0% (0%–0%) | 33% (0%–75%) | 33% (0%–75%) | 33% (0%–75%) |
| Reference ESI 1 answered 1–2 | 33% (0%–80%) | 33% (0%–80%) | 33% (0%–80%) | 0% (0%–0%) |
| Exact accuracy (all 30) | 30% (13%–47%) | 47% (30%–63%) | 50% (33%–67%) | 40% (23%–57%) |
| Under-triage (all 30) | 33% (17%–50%) | 37% (20%–53%) | 33% (17%–50%) | 43% (27%–60%) |
| Over-triage (all 30) | 37% (20%–53%) | 17% (3%–30%) | 17% (3%–30%) | 17% (3%–30%) |

The rule path uses the stricter floor (MODERATE → at most ESI 3), so the weights show more clearly here.

Scenarios whose answer differs between the three weightings: **6 of 30**.

| Scenario | Reference | Camera (posture / skin) | Visual 50/50 | 60/40 | 40/60 | LLM+floor 50/50 · 60/40 · 40/60 | Rules 50/50 · 60/40 · 40/60 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | 1 | chest / UNKNOWN | 0.5 HIGH | 0.6 HIGH | 0.4 MODERATE | 2 · 2 · 2 | 2 · 2 · 3 |
| S04 | 1 | chest / UNKNOWN | 0.5 HIGH | 0.6 HIGH | 0.4 MODERATE | 2 · 2 · 2 | 2 · 2 · 3 |
| S07 | 2 | chest / UNKNOWN | 0.5 HIGH | 0.6 HIGH | 0.4 MODERATE | 2 · 2 · 2 | 2 · 2 · 3 |
| S08 | 2 | chest / UNKNOWN | 0.5 HIGH | 0.6 HIGH | 0.4 MODERATE | 2 · 2 · 2 | 2 · 2 · 3 |
| S10 | 2 | head / UNKNOWN | 0.3 MODERATE | 0.36 MODERATE | 0.24 LOW | 4 · 4 · 4 | 3 · 3 · 4 |
| S11 | 2 | abdomen / NORMAL | 0.497 MODERATE | 0.538 HIGH | 0.457 MODERATE | 2 · 2 · 2 | 3 · 2 · 3 |

## 6. Which inputs are switched on — LLM + safety floor

| | Text only | Text + posture | Text + skin | Text + posture + skin |
| --- | --- | --- | --- | --- |
| Exact accuracy (in scope) | 42% (23%–61%) | 62% (42%–81%) | 46% (27%–66%) | 62% (42%–81%) |
| Within one level (in scope) | 88% (73%–100%) | 92% (78%–100%) | 88% (73%–100%) | 88% (72%–100%) |
| Weighted kappa (linear) (in scope) | 0.32 (0.08–0.55) | 0.59 (0.36–0.80) | 0.36 (0.12–0.59) | 0.56 (0.28–0.78) |
| Under-triage (in scope) | 33% (14%–52%) | 8% (0%–22%) | 29% (12%–48%) | 8% (0%–22%) |
| Over-triage (in scope) | 25% (9%–43%) | 29% (12%–48%) | 25% (9%–43%) | 29% (12%–48%) |
| ESI 5 answered as 5 | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) | 17% (0%–57%) |
| Reference ESI 1 answered 1–2 | 50% (0%–100%) | 67% (25%–100%) | 50% (0%–100%) | 67% (25%–100%) |
| Exact accuracy (all 30) | 33% (17%–50%) | 50% (33%–67%) | 37% (20%–53%) | 50% (33%–67%) |
| Under-triage (all 30) | 47% (30%–63%) | 27% (13%–43%) | 43% (27%–60%) | 27% (13%–43%) |
| Over-triage (all 30) | 20% (7%–33%) | 23% (10%–40%) | 20% (7%–33%) | 23% (10%–40%) |

## 7. How much each input moves the answer

Each scenario was answered with every combination of the three inputs switched on or off (8 combinations). The Shapley value gives each input its fair share of the change in the answer; the table shows the average size of that change in ESI levels and each input's share.

| | Patient's words | Posture | Skin |
| --- | --- | --- | --- |
| LLM alone — share | 53% (0.54 levels) | 43% (0.43 levels) | 4% (0.04 levels) |
| Whole system (LLM + floor) — share | 54% (0.57 levels) | 44% (0.46 levels) | 2% (0.02 levels) |
| LLM alone, scenarios where the camera reported a finding (n = 13) — share | 27% (0.40 levels) | 67% (1.00 levels) | 6% (0.09 levels) |
| Whole system, same scenarios — share | 30% (0.46 levels) | 68% (1.06 levels) | 2% (0.04 levels) |
| Scenarios whose answer changes when this input is removed (LLM alone / whole system) | 13 / 12 of 30 | 7 / 7 of 30 | 3 / 1 of 30 |

## 6b. Exploratory: the same weights when the skin part always answers

Added after the first round-B run (protocol amendment 2). The skin result comes from the round-A patch pipeline without the image check, so it answered for 29 of 30 scenarios (main system: 8). Posture and everything else are unchanged.


### LLM + safety floor, skin without the image check

| | Independent (no weights) | Posture 50 : Skin 50 | Posture 60 : Skin 40 | Posture 40 : Skin 60 |
| --- | --- | --- | --- | --- |
| Exact accuracy (in scope) | 42% (22%–62%) | 58% (38%–78%) | 38% (18%–58%) | 42% (23%–62%) |
| Within one level (in scope) | 75% (55%–92%) | 92% (78%–100%) | 75% (55%–92%) | 71% (50%–88%) |
| Weighted kappa (linear) (in scope) | 0.29 (0.10–0.48) | 0.58 (0.35–0.78) | 0.26 (0.08–0.45) | 0.27 (0.11–0.46) |
| Under-triage (in scope) | 4% (0%–14%) | 4% (0%–14%) | 4% (0%–14%) | 0% (0%–0%) |
| Over-triage (in scope) | 54% (33%–73%) | 38% (19%–57%) | 58% (38%–77%) | 58% (38%–77%) |
| ESI 5 answered as 5 | 0% (0%–0%) | 17% (0%–57%) | 0% (0%–0%) | 0% (0%–0%) |
| Reference ESI 1 answered 1–2 | 67% (25%–100%) | 67% (25%–100%) | 67% (25%–100%) | 67% (25%–100%) |
| Exact accuracy (all 30) | 33% (17%–50%) | 47% (30%–67%) | 30% (13%–47%) | 33% (17%–50%) |
| Under-triage (all 30) | 23% (10%–40%) | 23% (10%–40%) | 23% (10%–40%) | 20% (7%–37%) |
| Over-triage (all 30) | 43% (27%–60%) | 30% (13%–47%) | 47% (30%–63%) | 47% (30%–63%) |

### Rule-based path only, skin without the image check

| | Posture 50 : Skin 50 — rules only | Posture 60 : Skin 40 — rules only | Posture 40 : Skin 60 — rules only |
| --- | --- | --- | --- |
| Exact accuracy (in scope) | 42% (23%–62%) | 42% (22%–62%) | 46% (26%–65%) |
| Within one level (in scope) | 75% (55%–92%) | 75% (55%–92%) | 71% (50%–88%) |
| Weighted kappa (linear) (in scope) | 0.29 (0.10–0.49) | 0.30 (0.11–0.50) | 0.29 (0.12–0.50) |
| Under-triage (in scope) | 4% (0%–14%) | 4% (0%–14%) | 0% (0%–0%) |
| Over-triage (in scope) | 54% (35%–74%) | 54% (33%–74%) | 54% (35%–74%) |
| ESI 5 answered as 5 | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) |
| Reference ESI 1 answered 1–2 | 33% (0%–80%) | 33% (0%–80%) | 33% (0%–80%) |
| Exact accuracy (all 30) | 33% (17%–50%) | 33% (17%–50%) | 37% (20%–53%) |
| Under-triage (all 30) | 23% (10%–40%) | 23% (10%–40%) | 20% (7%–37%) |
| Over-triage (all 30) | 43% (27%–60%) | 43% (27%–60%) | 43% (27%–60%) |

| Input share, skin without the image check | Patient's words | Posture | Skin |
| --- | --- | --- | --- |
| LLM alone | 47% (0.61 levels) | 44% (0.57 levels) | 9% (0.12 levels) |
| Whole system | 42% (0.57 levels) | 48% (0.66 levels) | 9% (0.13 levels) |

## 8. Sensitivity: how strict the visual safety floor is

| Floor | Weights | Accuracy in scope | Under-triage in scope | Over-triage in scope | ESI 5 answered 5 | Reference ESI 1 answered 1–2 |
| --- | --- | --- | --- | --- | --- | --- |
| No visual floor | 50 : 50 | 54% | 17% | 29% | 17% | 67% |
| No visual floor | 60 : 40 | 62% | 8% | 29% | 17% | 67% |
| No visual floor | 40 : 60 | 62% | 8% | 29% | 17% | 67% |
| HIGH → at most ESI 2 (main) | 50 : 50 | 62% | 8% | 29% | 17% | 67% |
| HIGH → at most ESI 2 (main) | 60 : 40 | 62% | 8% | 29% | 17% | 67% |
| HIGH → at most ESI 2 (main) | 40 : 60 | 62% | 8% | 29% | 17% | 67% |
| HIGH → 2 and MODERATE → 3 | 50 : 50 | 62% | 8% | 29% | 17% | 67% |
| HIGH → 2 and MODERATE → 3 | 60 : 40 | 62% | 8% | 29% | 17% | 67% |
| HIGH → 2 and MODERATE → 3 | 40 : 60 | 62% | 8% | 29% | 17% | 67% |

## 9. Camera parts on the test photos

| | Posture correct | Skin: gave an answer | Skin: correct when answered | Truly flushed called flushed | Truly normal called flushed |
| --- | --- | --- | --- | --- | --- |
| Round A original (rules v1, whole-face CNN) | 16/30 | 30/30 | 16/30 | 0/14 | 0/16 |
| Round A fixed (rules v2, patches) | 24/30 | 29/30 | 17/29 | 10/14 | 8/16 |
| Round B (rules v3, patches + image check) | 25/30 | 8/30 | 5/8 | 2/14 | 0/16 |

Round B skin check gave no answer because: face too small in the image (12); image too blurry (5); result changes with image size (5).

## 10. Reliability of the main system

- LLM outputs that could not be used: 0 of 30 (none).
- Safety floor changed the LLM's answer in 3 of 30 scenarios.
- LLM time per scenario: median 11.9 s, 90th percentile 18.0 s, longest 36.5 s.
- The model's level field and its own step answers disagreed in 0 of 30.
