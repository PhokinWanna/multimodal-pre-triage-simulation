# M3 experiment scripts — run order

Protocol: [`docs/experiments/m3/PROTOCOL.md`](../../../docs/experiments/m3/PROTOCOL.md).
"kiosk env" = `.venv-kiosk` (requirements-kiosk.txt); "CNN env" = the TensorFlow environment (requirements-cnn.txt).

| # | Step | Command | Env |
| --- | --- | --- | --- |
| 1 | Convert the skin CNN and run the match test | `python scripts/experiments/m3/convert_skin_model.py` | CNN |
| 2 | Shortcut check on real patches (redness vs blur) | `python scripts/experiments/m3/skin_shortcut_check.py` | CNN |
| 3 | Collect 44 photos, fill `image_sources.csv`, then check | `python scripts/experiments/m3/validate_images.py` | either |
| 4 | Duplicate check against the training images | `python scripts/experiments/m3/check_duplicates.py` | either (needs Pillow) |
| 5 | Camera check on the 14 development photos | `python scripts/experiments/m3/check_dev_images.py` | kiosk |
| 6 | Tune on dev photos (with Claude), then freeze | — | — |
| 7 | Label the 30 scenarios yourself (`docs/experiments/m3/labels_phokin.csv`) | — | — |
| 8 | Compare labels, Ajarn decides disagreements | `python scripts/experiments/m3/compare_labels.py` | either |
| 9 | Smoke test (2 scenarios) — only when approved | `python scripts/experiments/m3/run_m3.py --mode A --limit 2` | kiosk + Ollama |
| 10 | **M3 runs** — only when approved | `run_m3.py --mode A`, `run_m3.py --mode B`, `run_m3.py --mode A --pipeline original` | kiosk + Ollama |
| 11 | Results | `python scripts/experiments/m3/analyze_m3.py --run-a <A dir> --run-b <B dir>` | CNN |

Send Claude the printed output of each step (and `results/experiments/m3/*.csv/json`).
