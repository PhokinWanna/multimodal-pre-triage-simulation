"""Compare Phokin's independent ESI labels with Claude's sealed draft and prepare the final labels.

    python scripts/experiments/m3/compare_labels.py

Fill docs/experiments/m3/labels_phokin.csv FIRST (without opening the sealed folder).
This prints the agreement (exact, weighted kappa) and writes
docs/experiments/m3/labels_final.csv with:
  * agreed cases filled in (status = agreed)
  * disagreements left for Ajarn Komsan (status = needs_decision, both proposals shown)
After Ajarn decides, type his level in esi_reference and set status = decided.
analyze_m3.py refuses to run while any row is not agreed/decided.
"""

from __future__ import annotations

import csv
import sys

import m3_metrics as M
from m3_common import LABELS_CLAUDE, LABELS_FINAL, LABELS_PHOKIN, write_csv


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["id"].strip(): r for r in csv.DictReader(f) if r.get("id")}


def main() -> int:
    phokin, claude = read(LABELS_PHOKIN), read(LABELS_CLAUDE)
    missing = [i for i, r in phokin.items() if not (r.get("esi") or "").strip()]
    if missing:
        print(f"labels_phokin.csv still has {len(missing)} empty row(s): {', '.join(missing)}")
        return 1
    ids = sorted(claude)
    a = [int(phokin[i]["esi"]) for i in ids]
    b = [int(claude[i]["esi"]) for i in ids]
    agree = sum(x == y for x, y in zip(a, b))
    print(f"Exact agreement: {agree}/{len(ids)} ({100 * agree / len(ids):.1f}%)")
    print(f"Weighted kappa: linear {M.weighted_kappa(a, b, 'linear'):.3f}, quadratic {M.weighted_kappa(a, b, 'quadratic'):.3f}")
    rows = []
    for i, x, y in zip(ids, a, b):
        if x == y:
            rows.append({"id": i, "esi_reference": x, "status": "agreed", "phokin": x, "claude": y, "note": ""})
        else:
            rows.append({"id": i, "esi_reference": "", "status": "needs_decision", "phokin": x, "claude": y,
                         "note": f"Claude: {claude[i]['rationale']} | Phokin: {phokin[i].get('notes', '')}"})
            print(f"  {i}: Phokin {x} vs Claude {y}  -> Ajarn decides")
    if LABELS_FINAL.exists():
        print(f"\n{LABELS_FINAL.name} already exists; not overwriting. Delete or rename it to regenerate.")
        return 0
    write_csv(LABELS_FINAL, rows, ["id", "esi_reference", "status", "phokin", "claude", "note"])
    print(f"\nWrote {LABELS_FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
