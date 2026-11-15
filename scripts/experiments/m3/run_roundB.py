"""Round B of the M3 experiment: the corrected system on the same 30 test scenarios.

Kiosk environment, Ollama running, after prompt_dev_eval.py:

    python scripts/experiments/m3/run_roundB.py --limit 2      # 1-minute smoke test (separate folder)
    python scripts/experiments/m3/run_roundB.py                # the real run (about 1-1.5 h on an M1)
    python scripts/experiments/m3/run_roundB.py --consistency  # optional: + 10 sampled runs of the main system
    python scripts/experiments/m3/run_roundB.py --no-qc        # + skin without the image check (sensitivity)

The run can be stopped (Ctrl+C) and started again with the same command: every LLM answer is
cached, so it continues where it stopped.

What is run (temperature 0, seed 0, one call per scenario and configuration):
  camera   round-B vision once per photo: posture rules v3 + skin CNN with the image check ("patch_qc")
  W50      MAIN SYSTEM: text + posture + skin, weighted fusion posture 50 % / skin 50 %
  W60/W40  the same with posture 60 / skin 40 and posture 40 / skin 60
  IND      posture and skin as two independent findings (the design up to round A), new prompt
  T, TP, TS         text only / text + posture / text + skin      (weighted 50/50, part switched off)
  NONE, P, S, PS    the same four without the patient's words     (for the input-contribution analysis)
  *_rules  W50, W60, W40, IND decided by the rule-based path only (no LLM; instant)
Output: results/experiments/m3/roundB/run_main/ (records.jsonl, vision.json, config.json, llm_cache.jsonl).
Then: python scripts/experiments/m3/analyze_roundB.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

from m3_common import IMAGES_DIR, PROJECT_ROOT, SCENARIOS_CSV, sharpness
from roundb_common import (ROUNDB_DIR, RUN_DIR, SELECTION_JSON, intake_from_text, make_engine, ollama_ready, rel,
                           sha256_file, utf8_console)
from run_m3 import git_commit, pair_images, read_scenarios

NEW_MODEL = PROJECT_ROOT / "models" / "skin_classifier.tflite"
PRIMARY = "W50"
# name, text, posture, skin, fusion mode, (w_posture, w_skin)
LLM_CONFIGS = [
    ("W50", True, True, True, "weighted", (0.5, 0.5)),
    ("IND", True, True, True, "independent", (0.5, 0.5)),
    ("W60", True, True, True, "weighted", (0.6, 0.4)),
    ("W40", True, True, True, "weighted", (0.4, 0.6)),
    ("T", True, False, False, "weighted", (0.5, 0.5)),
    ("TP", True, True, False, "weighted", (0.5, 0.5)),
    ("TS", True, False, True, "weighted", (0.5, 0.5)),
    ("NONE", False, False, False, "weighted", (0.5, 0.5)),
    ("P", False, True, False, "weighted", (0.5, 0.5)),
    ("S", False, False, True, "weighted", (0.5, 0.5)),
    ("PS", False, True, True, "weighted", (0.5, 0.5)),
]
RULES_CONFIGS = [(f"{n}_rules", t, p, s, m, w) for n, t, p, s, m, w in LLM_CONFIGS if n in ("W50", "IND", "W60", "W40")]
# Exploratory sensitivity analysis (amendment 2, added after the first round-B run): the same weights
# when the skin result comes from the round-A patch pipeline WITHOUT the image check, i.e. the skin
# part always answers. Only configurations that use the skin result are repeated.
NOQC = "_noQC"
NOQC_CONFIGS = [(n + NOQC, t, p, s, m, w) for n, t, p, s, m, w in LLM_CONFIGS if s and n in ("W50", "IND", "W60", "W40", "TS", "S", "PS")]
NOQC_RULES = [(n + NOQC, t, p, s, m, w) for n, t, p, s, m, w in RULES_CONFIGS if n != "IND_rules"]
CONSISTENCY = {"temperature": 0.8, "seeds": list(range(1, 11))}


def analyse_photo(vision, path: Path) -> dict:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Cannot read photo {path}")
    a = vision.analyze_image(img)
    s = a["skin"]
    return {"person_found": a["person_found"], "face_found": a["face_found"], "visual_signs": a["visual_signs"],
            "posture_label": a["posture"].label() if a["posture"] else "none",
            "posture_filters": a["posture"].filters if a["posture"] else [],
            "skin_status": s.status, "skin_probability": s.probability, "skin_reason": s.reason,
            "skin_patches": s.patch_probabilities, "skin_quality": s.quality, "skin_method": s.method}


def camera_results(scenarios: list[dict], out: Path, vision=None, skin_pipeline: str = "patch_qc") -> dict:
    """Round-B camera analysis of every test photo, computed once and stored (photos are hashed)."""
    files = {}
    for sc in scenarios:
        for f in (sc["image_file"], sc.get("image_file_skin") or ""):
            if f:
                files[f] = sha256_file(IMAGES_DIR / "test" / f)
    if out.is_file():
        stored = json.loads(out.read_text(encoding="utf-8"))
        if stored.get("photo_sha256") == files and stored.get("skin_pipeline") == skin_pipeline:
            return stored
    if vision is None:
        from triage_kiosk.vision.engine import VisionEngine
        vision = VisionEngine(static_image_mode=True, posture_version="v3", skin_pipeline=skin_pipeline,
                              skin_model_path=str(NEW_MODEL), require_skin_model=True)
    cases = {}
    for sc in scenarios:
        body = IMAGES_DIR / "test" / sc["image_file"]
        cam = analyse_photo(vision, body)
        skin_file = sc.get("image_file_skin") or ""
        skin_path = IMAGES_DIR / "test" / skin_file if skin_file else body
        if skin_file:      # two-photo case: posture from the body photo, skin from the face photo
            face = analyse_photo(vision, skin_path)
            for k in ("skin_status", "skin_probability", "skin_reason", "skin_patches", "skin_quality",
                      "skin_method", "face_found"):
                cam[k] = face[k]
        cam.update({"image_file": sc["image_file"], "image_file_skin": skin_file,
                    "sharpness_posture_photo": sharpness(body), "sharpness_skin_photo": sharpness(skin_path)})
        cases[sc["id"]] = cam
    skin_model = getattr(vision, "skin_model", None)
    stored = {"created": dt.datetime.now().isoformat(timespec="seconds"),
              "posture_rules": getattr(vision, "posture_version", "?"), "skin_pipeline": getattr(vision, "skin_pipeline", "?"),
              "skin_model": None if skin_model is None else {"name": skin_model.info.name, "threshold": skin_model.info.threshold},
              "photo_sha256": files, "cases": cases}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stored, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return stored


def one_call(engine, sc: dict, cam: dict, cfg: tuple, selection: dict, timeout: float, use_llm: bool = True,
             temperature: float = 0.0, seed: int = 0, run: int = 0) -> dict:
    name, text_on, posture_on, skin_on, mode, weights = cfg
    text = sc["patient_text"] if text_on else ""
    conversation = [{"role": "patient", "text": text}] if text else []
    t0 = time.time()
    engine.last_call = {}
    v = engine.evaluate_final_case(
        conversation, cam["visual_signs"] if posture_on else None,
        cam["skin_status"] if skin_on else "NOT_ASSESSED", "NORMAL_COMPLETE", sc.get("language", "en"),
        temperature=temperature, seed=seed, timeout=timeout, use_llm=use_llm, intake=intake_from_text(text),
        skin_probability=cam["skin_probability"] if skin_on else None,
        skin_reason=cam["skin_reason"] if skin_on else "switched off",
        prompt_version=selection["prompt_version"], fusion_mode=mode, weights=weights, cap_policy="high_only",
        esi_from_steps=selection["esi_from_steps"])
    d = v["decision"]
    visual = d["visual"] or {}
    return {"id": sc["id"], "config": name, "run": run, "seed": seed, "temperature": temperature,
            "text_on": text_on, "posture_on": posture_on, "skin_on": skin_on, "fusion_mode": mode,
            "w_posture": weights[0], "w_skin": weights[1], "llm_used": use_llm,
            "visual_score": visual.get("score"), "visual_band": visual.get("band"),
            "esi_llm": d["esi_llm"], "esi_llm_level_field": d["esi_llm_level_field"],
            "esi_llm_from_steps": d["esi_llm_from_steps"], "llm_valid": d["llm_valid"],
            "fallback_reason": d["fallback_reason"], "esi_rules": d["esi_rules"],
            "esi_before_floor": d["esi_before_floor"], "caps": d["caps"], "esi_final": d["esi_final"],
            "scope": v.get("scope"), "urgency": v.get("urgency"),
            "llm_latency_s": engine.last_call.get("latency"), "cached": bool(engine.last_call.get("cached")),
            "wall_s": round(time.time() - t0, 3),
            "steps": {k: v.get(k) for k in ("step_a", "step_b", "resources")},
            "soap": {k: v.get(k) for k in ("subjective", "objective", "assessment", "plan")}}


def run(args, llm=None, vision=None, out_dir: Path | None = None, selection: dict | None = None) -> int:
    utf8_console()
    if selection is None:
        if not SELECTION_JSON.is_file():
            print("Run prompt_dev_eval.py first: the prompt has not been chosen on the development scenarios yet.")
            return 1
        selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))["selected"]
    model_info = "FAKE (test)"
    if llm is None:
        ok, model_info = ollama_ready()
        if not ok:
            print(model_info)
            return 1
    scenarios = pair_images(read_scenarios(args.scenarios))
    out = out_dir or (ROUNDB_DIR / "run_smoke" if args.limit else RUN_DIR)
    if args.limit:
        scenarios = scenarios[: args.limit]
        shutil.rmtree(out, ignore_errors=True) if out_dir is None else None
    out.mkdir(parents=True, exist_ok=True)

    key = {"selection": selection, "scenarios_sha256": sha256_file(args.scenarios), "n_scenarios": len(scenarios)}
    cfg_path = out / "config.json"
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in key} != key:
            print(f"{rel(out)} holds a run made with another prompt selection or scenario file. "
                  "Move or rename that folder first; results are never mixed.")
            return 1
    from triage_kiosk.config import SystemConfig
    cam_all = camera_results(scenarios, out / "vision.json", vision)
    cam_noqc = camera_results(scenarios, out / "vision_noQC.json", vision, skin_pipeline="patch") if args.no_qc else None
    config = {**key, "created": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
              "llm": model_info, "llm_model": SystemConfig.OLLAMA_MODEL, "llm_timeout_s": args.timeout,
              "primary_config": PRIMARY, "temperature": 0.0, "seed": 0,
              "configs": [dict(zip(("name", "text", "posture", "skin", "fusion_mode", "weights"), c)) for c in LLM_CONFIGS],
              "rules_configs": [c[0] for c in RULES_CONFIGS], "cap_policy": "high_only",
              "missing_policy": SystemConfig.FUSION_MISSING,
              "posture_rules": cam_all["posture_rules"], "skin_pipeline": cam_all["skin_pipeline"],
              "skin_model": cam_all["skin_model"], "consistency": CONSISTENCY if args.consistency else None,
              "noqc_sensitivity": bool(args.no_qc)}
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    engine, cache = make_engine(out / "llm_cache.jsonl", llm=llm)
    print(f"Prompt {selection['candidate']}; {len(scenarios)} scenarios; output {rel(out)}")
    print("Camera (round B): " + ", ".join(f"{sid}={c['posture_label']}/{c['skin_status']}" for sid, c in cam_all["cases"].items()))
    records, failures, t_start, done_live = [], 0, time.time(), 0
    jobs = [(cfg, True, 0.0, 0, 0) for cfg in LLM_CONFIGS]
    if args.consistency:
        primary = next(c for c in LLM_CONFIGS if c[0] == PRIMARY)
        jobs += [((f"{PRIMARY}_sampled",) + primary[1:], True, CONSISTENCY["temperature"], s, i + 1)
                 for i, s in enumerate(CONSISTENCY["seeds"])]
    jobs += [(cfg, False, 0.0, 0, 0) for cfg in RULES_CONFIGS]
    if args.no_qc:
        jobs += [(cfg, True, 0.0, 0, 0) for cfg in NOQC_CONFIGS] + [(cfg, False, 0.0, 0, 0) for cfg in NOQC_RULES]
    total = len(jobs) * len(scenarios)
    try:
        for j, (cfg, use_llm, temp, seed, run_no) in enumerate(jobs):
            line = []
            for sc in scenarios:
                cams = cam_noqc if cfg[0].endswith(NOQC) else cam_all
                rec = one_call(engine, sc, cams["cases"][sc["id"]], cfg, selection, args.timeout, use_llm, temp, seed, run_no)
                records.append(rec)
                if rec["fallback_reason"] and any(x in rec["fallback_reason"] for x in ("TIMEOUT", "UNREACHABLE", "HTTP")):
                    failures += 1
                if use_llm and not rec["cached"]:
                    done_live += 1
                line.append(f"{rec['esi_final']}")
            left = total - len(records)
            per = (time.time() - t_start) / max(1, done_live)
            eta = f", about {left * per / 60:.0f} min left if nothing is cached" if done_live and left else ""
            label = cfg[0] + (f" run {run_no}" if run_no else "")
            print(f"[{j + 1}/{len(jobs)}] {label:<14} final ESI: {' '.join(line)}{eta}")
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue (finished answers are kept).")
        return 130
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nLLM calls: {cache.misses} new, {cache.hits} from the cache. Saved {rel(out / 'records.jsonl')}")
    if failures:
        print(f"{failures} call(s) got no answer (timeout or Ollama stopped). Run the same command again to repeat only those.")
        return 2
    print("Complete." + ("" if args.limit else " Next: python scripts/experiments/m3/analyze_roundB.py"))
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds per LLM call")
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS_CSV)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenarios, into run_smoke/ (smoke test)")
    ap.add_argument("--consistency", action="store_true", help="also 10 sampled runs (temperature 0.8) of the main system")
    ap.add_argument("--no-qc", action="store_true",
                    help="also repeat the skin configurations with the round-A skin pipeline (no image check)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
