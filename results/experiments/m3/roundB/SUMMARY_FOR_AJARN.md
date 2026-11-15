# M3 round B — results summary (2026-11-15)

Same 30 scenarios, 37 photos and reference labels as round A. Local LLM: Llama 3.2 3B (Ollama), temperature 0.
Reference ESI 1 (6 cases) = **out of the kiosk's scope** → scored as "sent to staff"; main figures use the
**24 in-scope cases (ESI 2–5)**. Brackets = 95 % bootstrap intervals. Full tables: `analysis/report.md`.

## What changed between round A and round B (tuned on development data only)

| Part | Round A (fixed) | Round B |
| --- | --- | --- |
| Posture rules | v2 | v3 (chest zone re-fitted on ~1,970 development images; hold-out chest sensitivity 67 % → 75 %) |
| Skin | CNN on face patches, always answers | + image check: no answer if the face is too small, blurred, too dark/bright, or the answer changes with image size |
| Camera fusion | posture and skin separate | weighted visual score = wP·posture + wS·skin; HIGH (≥ 0.50) → at least ESI 2 |
| Prompt | v1 | v1 kept: the new step-by-step ESI prompts (v2, v3) were worse on the development cases (kappa 0.03 / 0.33 vs 0.36) — the 3B model answered "no / no / no resources" for almost everything |
| Robustness | — | a cut-off LLM answer is asked again once (0 unusable answers in round B) |

## 1. Main result (in scope, n = 24)

| | Round A original | Round A fixed | **Round B** |
| --- | --- | --- | --- |
| Exact accuracy | 33 % | 50 % | **62 %** (42–81) |
| Weighted kappa (linear) | 0.22 | 0.47 | **0.56** (0.28–0.78) |
| Under-triage (less urgent than reference) | 46 % | 25 % | **8 %** (0–22) |
| Over-triage | 21 % | 25 % | 29 % (12–48) |
| ESI 2 recognised (answered 1–2) | 0 % | 50 % | **67 %** |
| ESI 5 answered as 5 | 17 % | 17 % | 17 % |

All 30 cases: accuracy 27 % → 40 % → **50 %**; under-triage 57 % → 40 % → **27 %**.
The intervals overlap: with 24–30 cases the improvement is a trend, not a proven difference.

## 2. Ajarn's point 2.1 — ESI 1 out of scope, ESI 5 included

- **ESI 1 (n = 6):** never answered as ESI 1 (0 %) in any round. Answered ESI 1–2 (urgent nurse alert) in 4/6 (67 %; round A fixed 5/6).
- **ESI 5 (n = 6):** 1/6 answered as 5 in every round.
- Reason: the 3B model rarely uses the two ends of the scale (it answers 2–4). Giving it the ESI decision steps did not help; it made it worse. **The kiosk's out-of-scope detection therefore relies on the emergency keywords during the interview and the urgent alert for ESI 1–2, not on the final LLM level.** A larger model or rule-based resource counting is needed for ESI 1/5 → future work.

## 3. Ajarn's point 2.2 — posture : skin weights

**Main system (skin with the image check):** the three weightings gave **identical results** (62 % accuracy, kappa 0.56). Reason: the image check refused 22 of 30 test photos (face too small in full-body photos 12, blurred 5, size-dependent answer 5), so skin almost never contributed and only posture moved the score.

**Exploratory (skin without the image check, answers 29/30, but calls 8/16 normal faces "flushed"):**

| In scope (n = 24) | Independent | 50 : 50 | 60 : 40 | 40 : 60 |
| --- | --- | --- | --- | --- |
| Exact accuracy | 42 % | **58 %** | 38 % | 42 % |
| Weighted kappa | 0.29 | **0.58** | 0.26 | 0.27 |
| Under-triage | 4 % | 4 % | 4 % | 0 % |
| Over-triage | 54 % | 38 % | 58 % | 58 % |
| Rules only (no LLM): accuracy | — | 42 % | 42 % | 46 % |

- 50 : 50 was best with the LLM, but the pattern is **not monotonic** (both 60:40 and 40:60 are worse), and without the LLM the weights barely matter (42–46 %). So the differences come mostly from the LLM reacting to the numbers written in its prompt, not from a real weighting effect.
- Using the unreliable skin result raises over-triage strongly (29 % → 38–58 %). **Conclusion: with the current skin CNN, no weighting is better than letting the image check switch skin off; the skin part needs a better model or better photos before its weight matters.**

## 4. Does the LLM use the inputs equally? (my question to Ajarn)

Every case was answered with all 8 combinations of words / posture / skin switched on or off; the Shapley value
splits the change in the answer fairly between the three inputs.

| Share of the change in the ESI answer | Patient's words | Posture | Skin |
| --- | --- | --- | --- |
| All 30 cases, LLM alone | **53 %** | **43 %** | 4 % |
| Cases where the camera found something (n = 13) | 27 % | **67 %** | 6 % |
| Skin without image check (exploratory) | 47 % | 44 % | 9 % |
| Cases whose answer changes when the input is removed | 13 / 30 | 7 / 30 | 3 / 30 |

**Answer: the inputs are not weighted equally.** The patient's words and posture carry almost all of the decision; skin
moves the answer in very few cases, even when it always answers. When posture is present, it dominates.

## 5. Camera parts (test photos)

| | Posture correct | Skin answered | Skin correct when answered | Flushed found |
| --- | --- | --- | --- | --- |
| Round A original | 16/30 | 30/30 | 16/30 | 0/14 |
| Round A fixed | 24/30 | 29/30 | 17/29 | 10/14 (8/16 normal called flushed) |
| Round B | 25/30 | 8/30 | 5/8 | 2/14 (0/16 normal called flushed) |

## 6. Reliability

0 unusable LLM answers; the safety floor raised the urgency in 3/30 cases; LLM time median 11.9 s (90th percentile 18 s) on an M1 MacBook Air.

## 7. Limitations

Same 30 cases as round A (the fixes were tuned on development data, but the choice of what to fix came from round-A errors); n = 30; stock photos and two-photo cases; reference labels AI-assisted and reviewed by the researcher, not by triage nurses; text input skips speech recognition.

## 8. Phase III — future work (from Ajarn's comments)

1. **Online / web application** — the nurse dashboard is already a web app (FastAPI); next is a browser kiosk client so the system runs without installing Python.
2. **Image input pre-processing: still photo vs video** — round B's image check (face size, blur, lighting, size-consistency) is a first version. With video, the kiosk can pick the sharpest, best-lit frame over a few seconds and accept a posture only if it is held (the live kiosk already requires 3 s), which should reduce the 22/30 "no skin answer" seen with single photos.

Figures (`analysis/`): `fig_rounds`, `fig_confusion`, `fig_recall`, `fig_weights`, `fig_weights_noQC`, `fig_weights_rules`, `fig_modalities`, `fig_contribution`, `fig_contribution_noQC` (PNG + PDF).
