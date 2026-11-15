"""Choose the final-verdict prompt on the DEVELOPMENT scenarios (round B, never the test cases).

    python scripts/experiments/m3/prompt_dev_eval.py          # kiosk environment, Ollama running (~8 min)

For each of the 15 development scenarios (docs/experiments/m3/dev_scenarios.csv: patient text +
a scripted camera result) the LLM is asked once per prompt (v1, v2, v3; temperature 0, seed 0,
weighted fusion 50/50). Five candidates are scored on the FINAL system output (LLM + safety floor):

    v1          prompt used up to round A
    v2-level    ESI decision steps; level = the model's "esi_level" field
    v2-steps    ESI decision steps; level = implied by the model's own step answers
    v3-level    v2 + worked examples; "esi_level" field
    v3-steps    v2 + worked examples; step answers

Selection rule (fixed in docs/experiments/m3/PROTOCOL.md before this script was run):
  1. highest linearly weighted kappa against the development reference levels;
  2. if within 0.01: fewer under-triaged scenarios;  3. then fewer invalid LLM outputs;
  4. then the simpler candidate (the order above).
The choice is written to results/experiments/m3/roundB/prompt_selection.json and used by
run_roundB.py. It is not redone once the test run exists (use --force only for a documented reason).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

import m3_metrics as M
from roundb_common import (CANDIDATES, DEV_SCENARIOS_CSV, ROUNDB_DIR, RUN_DIR, SELECTION_JSON, candidate_name,
                           final_level, intake_from_text, make_engine, ollama_ready, read_csv, rel, sha256_file,
                           signs_from_label, utf8_console)

PROMPTS = ("v1", "v2", "v3")


def run_dev(engine, scenarios: list[dict], timeout: float, log=print) -> list[dict]:
    rows = []
    for n, sc in enumerate(scenarios, 1):
        prob = float(sc["camera_skin_probability"]) if sc["camera_skin_probability"] else None
        for pv in PROMPTS:
            t0 = time.time()
            v = engine.evaluate_final_case(
                [{"role": "patient", "text": sc["patient_text"]}], signs_from_label(sc["camera_posture"]),
                sc["camera_skin_status"], "NORMAL_COMPLETE", sc["language"], temperature=0.0, seed=0,
                timeout=timeout, intake=intake_from_text(sc["patient_text"]), skin_probability=prob,
                prompt_version=pv, fusion_mode="weighted", weights=(0.5, 0.5), cap_policy="high_only",
                esi_from_steps=False)
            d = v["decision"]
            rows.append({"id": sc["id"], "ref": int(sc["esi_reference"]), "prompt": pv,
                         "level_field": d["esi_llm_level_field"], "steps": d["esi_llm_from_steps"],
                         "llm_valid": d["llm_valid"], "fallback_reason": d["fallback_reason"],
                         "esi_rules": d["esi_rules"], "caps": d["caps"],
                         "visual_band": (d["visual"] or {}).get("band"),
                         "latency_s": engine.last_call.get("latency"), "cached": engine.last_call.get("cached"),
                         "wall_s": round(time.time() - t0, 2)})
        last = rows[-len(PROMPTS):]
        log(f"[{n}/{len(scenarios)}] {sc['id']} ref {sc['esi_reference']}: "
            + "  ".join(f"{r['prompt']} level={r['level_field']} steps={r['steps']}" for r in last))
    return rows


def score(rows: list[dict]) -> list[dict]:
    table = []
    for order, (pv, from_steps) in enumerate(CANDIDATES):
        mine = [r for r in rows if r["prompt"] == pv]
        ref, pred, invalid = [], [], 0
        for r in mine:
            llm = r["steps"] if (from_steps and r["steps"] is not None) else r["level_field"]
            if llm is None:
                llm = r["steps"]
            if llm is None:                      # no usable LLM answer: the system falls back to the rules
                invalid += 1
                llm = r["esi_rules"]
            ref.append(r["ref"])
            pred.append(final_level(llm, r["caps"]))
        s = M.summary(ref, pred)
        table.append({"candidate": candidate_name(pv, from_steps), "prompt_version": pv, "esi_from_steps": from_steps,
                      "order": order, "n": len(ref), "kappa_linear": round(s["kappa_linear"], 4),
                      "accuracy": round(s["accuracy"], 4),
                      "under_triaged": int(round(s["under_triage"] * len(ref))),
                      "over_triaged": int(round(s["over_triage"] * len(ref))),
                      "invalid_llm_outputs": invalid, "predictions": dict(zip([r["id"] for r in mine], pred))})
    return table


def select(table: list[dict]) -> dict:
    best_kappa = max(t["kappa_linear"] for t in table)
    near = [t for t in table if best_kappa - t["kappa_linear"] < 0.01]
    return min(near, key=lambda t: (t["under_triaged"], t["invalid_llm_outputs"], t["order"]))


def main(argv=None, llm=None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--force", action="store_true", help="redo the selection even if one exists")
    args = ap.parse_args(argv)

    if SELECTION_JSON.is_file() and not args.force:
        sel = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
        print(f"A selection already exists: {sel['selected']['candidate']} (made {sel['created']}). Nothing to do.")
        return 0
    if (RUN_DIR / "records.jsonl").is_file() and not args.force:
        print("The round-B test run already exists; the prompt is not re-selected after it.")
        return 1
    model_info = "FAKE (test)"
    if llm is None:
        ok, model_info = ollama_ready()
        if not ok:
            print(model_info)
            return 1
    scenarios = read_csv(DEV_SCENARIOS_CSV)
    engine, cache = make_engine(ROUNDB_DIR / "cache" / "llm_cache_dev.jsonl", llm=llm)
    print(f"{len(scenarios)} development scenarios x {len(PROMPTS)} prompts (answers already cached are reused)\n")
    rows = run_dev(engine, scenarios, args.timeout)
    failed = [r for r in rows if r["fallback_reason"] and any(x in r["fallback_reason"] for x in ("TIMEOUT", "UNREACHABLE", "HTTP"))]
    if failed:
        print(f"\n{len(failed)} call(s) did not get an answer (timeout / Ollama not reachable). "
              "Run the same command again; finished answers are kept.")
        return 1
    table = score(rows)
    chosen = select(table)
    print(f"\n{'candidate':<10}{'kappa':>8}{'accuracy':>10}{'under':>7}{'over':>6}{'invalid':>9}")
    for t in table:
        mark = "  <- selected" if t is chosen else ""
        print(f"{t['candidate']:<10}{t['kappa_linear']:>8.3f}{t['accuracy'] * 100:>9.0f}%{t['under_triaged']:>7}"
              f"{t['over_triaged']:>6}{t['invalid_llm_outputs']:>9}{mark}")
    lat = {pv: [r["latency_s"] for r in rows if r["prompt"] == pv and r["latency_s"]] for pv in PROMPTS}
    out = {"created": dt.datetime.now().isoformat(timespec="seconds"), "model": model_info,
           "dev_scenarios_sha256": sha256_file(DEV_SCENARIOS_CSV), "settings": "temperature 0, seed 0, weighted 50/50, floor high_only",
           "rule": "max linear kappa; within 0.01 -> fewer under-triaged -> fewer invalid -> simpler",
           "selected": {k: chosen[k] for k in ("candidate", "prompt_version", "esi_from_steps")},
           "table": table, "median_latency_s": {pv: (sorted(v)[len(v) // 2] if v else None) for pv, v in lat.items()},
           "calls": rows}
    SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSelected: {chosen['candidate']}. Saved {rel(SELECTION_JSON)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
