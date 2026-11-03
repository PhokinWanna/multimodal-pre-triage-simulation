"""M3 replay: test photo + patient text -> camera analysis -> LLM final verdict (ESI 1-5).

Kiosk environment (Ollama running with llama3.2:latest). DO NOT run until the experiment
is approved. Typical use:

    # main result (option A): one deterministic run per scenario, corrected pipeline
    python scripts/experiments/m3/run_m3.py --mode A --pipeline fixed
    # supporting result (option B): 10 runs per scenario with sampling
    python scripts/experiments/m3/run_m3.py --mode B --pipeline fixed
    # appendix: the original kiosk pipeline on the same cases
    python scripts/experiments/m3/run_m3.py --mode A --pipeline original

Settings fixed in advance (pre-registered, see docs/experiments/m3/PROTOCOL.md):
  * LLM: SystemConfig.OLLAMA_MODEL with the kiosk's own final-verdict prompt.
  * Mode A: temperature 0, 1 run, seed 0.   Mode B: temperature 0.8, 10 runs, seeds 1..10.
  * Timeout 60 s per LLM call (the real kiosk uses 6 s; latency is recorded and the number of
    calls that would have exceeded 6 s is reported).
  * Camera: single-frame analysis (no 3-second hold), static MediaPipe.
  * "fixed" = posture rules v2 + forehead/cheek patches + converted SkinCNN-FL-α0.75 (validation threshold).
    "original" = posture rules v1 + whole-face crop + original kiosk model (threshold 0.65).

Output: results/experiments/m3/runs/<time>_<pipeline>_<mode>/ with cases.jsonl (one line per
LLM call), vision.csv (camera results per scenario) and config.json (everything needed to repeat).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from m3_common import (IMAGES_DIR, PROJECT_ROOT, RESULTS_DIR, SCENARIOS_CSV, case_key, is_two_photo, photo_role,
                       read_sources, sharpness, write_csv)

MODE_SETTINGS = {"A": {"temperature": 0.0, "runs": 1, "seed0": 0}, "B": {"temperature": 0.8, "runs": 10, "seed0": 1}}
KIOSK_TIMEOUT_S = 6.0
LEGACY_MODEL = PROJECT_ROOT / "models" / "legacy" / "skin_classifier_original.tflite"
NEW_MODEL = PROJECT_ROOT / "models" / "skin_classifier.tflite"


def read_scenarios(path: Path = SCENARIOS_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    return [r for r in rows if r.get("id")]


def pair_images(scenarios: list[dict]) -> list[dict]:
    """Use image_file (and image_file_skin) when given; otherwise take the next unused test photo(s).

    Single-photo categories: one photo gives posture AND skin.
    Two-photo "both_*" categories: <case>_body gives the posture, <case>_face gives the skin result.
    """
    singles: dict[str, list[str]] = {}
    pairs: dict[str, dict[str, dict[str, str]]] = {}
    for rec in read_sources():
        if rec.set != "test":
            continue
        if is_two_photo(rec.category):
            pairs.setdefault(rec.category, {}).setdefault(case_key(rec.file), {})[photo_role(rec.file)] = rec.file
        else:
            singles.setdefault(rec.category, []).append(rec.file)
    used = {s.get("image_file") for s in scenarios if s.get("image_file")}
    for s in scenarios:
        if s.get("image_file"):
            continue
        cat = s["image_category"]
        if is_two_photo(cat):
            pool = [k for k, roles in sorted(pairs.get(cat, {}).items())
                    if {"body", "face"} <= set(roles) and roles["body"] not in used]
            if not pool:
                raise SystemExit(f"No unused body+face pair left for scenario {s['id']} (category {cat})")
            roles = pairs[cat][pool[0]]
            s["image_file"], s["image_file_skin"] = roles["body"], roles["face"]
        else:
            pool = [f for f in sorted(singles.get(cat, [])) if f not in used]
            if not pool:
                raise SystemExit(f"No unused test photo left for scenario {s['id']} (category {cat})")
            s["image_file"] = pool[0]
        used.add(s["image_file"])
    return scenarios


def valid_esi(value) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def ollama_info(host: str, model: str) -> dict:
    import requests
    try:
        tags = requests.get(f"{host}/api/tags", timeout=5).json().get("models", [])
    except Exception as e:
        raise SystemExit(f"Ollama is not reachable at {host} ({e}). Start it with: ollama serve")
    match = [m for m in tags if m.get("name") == model or m.get("model") == model]
    if not match:
        raise SystemExit(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
    return {"name": model, "digest": match[0].get("digest"), "size": match[0].get("size")}


def build_vision(pipeline: str):
    from triage_kiosk.vision.engine import VisionEngine
    if pipeline == "fixed":
        if not NEW_MODEL.with_suffix(".json").is_file():
            raise SystemExit("models/skin_classifier.json missing: run convert_skin_model.py first")
        return VisionEngine(static_image_mode=True, posture_version="v2", skin_pipeline="patch",
                            skin_model_path=str(NEW_MODEL), require_skin_model=True)
    model = LEGACY_MODEL if LEGACY_MODEL.is_file() else None
    if model is None:
        raise SystemExit("models/legacy/skin_classifier_original.tflite missing (created by convert_skin_model.py)")
    return VisionEngine(static_image_mode=True, posture_version="v1", skin_pipeline="legacy_whole_face",
                        skin_model_path=str(model), require_skin_model=True)


def analyse_photo(vision, image_path: Path) -> dict:
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Cannot read photo {image_path}")
    a = vision.analyze_image(img)
    s = a["skin"]
    return {
        "person_found": a["person_found"], "face_found": a["face_found"],
        "visual_signs": a["visual_signs"],
        "posture_label": a["posture"].label() if a["posture"] else "none",
        "posture_filters": a["posture"].filters if a["posture"] else [],
        "posture_debug": a["posture"].debug if a["posture"] else {},
        "skin_status": s.status, "skin_probability": s.probability, "skin_patches": s.patch_probabilities,
        "skin_method": s.method,
    }


def run(args, llm_call: Optional[Callable] = None, vision=None) -> Path:
    from triage_kiosk.config import SystemConfig
    settings = MODE_SETTINGS[args.mode]
    runs = args.runs or settings["runs"]
    scenarios = pair_images(read_scenarios(args.scenarios))
    if args.limit:
        scenarios = scenarios[: args.limit]

    if llm_call is None:
        from triage_kiosk.triage.reasoning import ReasoningEngine
        llm_info = ollama_info(SystemConfig.OLLAMA_HOST, SystemConfig.OLLAMA_MODEL)
        engine = ReasoningEngine()
        engine.warmup()
        llm_call = engine.evaluate_final_case
    else:
        llm_info = {"name": "FAKE (test run)"}
    vision = vision or build_vision(args.pipeline)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out or RESULTS_DIR / "runs") / f"{stamp}_{args.pipeline}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    skin_model = getattr(vision, "skin_model", None)
    config = {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "mode": args.mode, "pipeline": args.pipeline, "runs_per_scenario": runs,
        "temperature": settings["temperature"], "seeds": [settings["seed0"] + i for i in range(runs)],
        "llm_timeout_s": args.timeout, "kiosk_timeout_s": KIOSK_TIMEOUT_S, "llm": llm_info,
        "posture_rules": getattr(vision, "posture_version", "?"), "skin_pipeline": getattr(vision, "skin_pipeline", "?"),
        "skin_model": None if skin_model is None else {"name": skin_model.info.name, "threshold": skin_model.info.threshold},
        "modalities": {"posture": not args.no_posture, "skin": not args.no_skin},
        "scenarios_sha256": hashlib.sha256(Path(args.scenarios).read_bytes()).hexdigest(),
        "n_scenarios": len(scenarios),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    vision_rows = []
    with open(out / "cases.jsonl", "w", encoding="utf-8") as jl:
        for n, sc in enumerate(scenarios, 1):
            cam = analyse_photo(vision, IMAGES_DIR / "test" / sc["image_file"])
            skin_file = sc.get("image_file_skin") or ""
            if skin_file:   # two-photo case: posture from the body photo, skin from the face photo
                face = analyse_photo(vision, IMAGES_DIR / "test" / skin_file)
                for k in ("skin_status", "skin_probability", "skin_patches", "skin_method", "face_found"):
                    cam[k] = face[k]
            signs = cam["visual_signs"] if not args.no_posture else {k: False for k in cam["visual_signs"]}
            skin_status = cam["skin_status"] if not args.no_skin else "NOT_ASSESSED"
            body_path = IMAGES_DIR / "test" / sc["image_file"]
            skin_path = IMAGES_DIR / "test" / skin_file if skin_file else body_path
            vision_rows.append({"id": sc["id"], "image_file": sc["image_file"], "image_file_skin": skin_file,
                                "sharpness_posture_photo": sharpness(body_path) if body_path.is_file() else None,
                                "sharpness_skin_photo": sharpness(skin_path) if skin_path.is_file() else None,
                                "posture_label": cam["posture_label"],
                                "skin_status": cam["skin_status"], "skin_probability": cam["skin_probability"],
                                "person_found": cam["person_found"], "face_found": cam["face_found"],
                                "skin_patches": json.dumps(cam["skin_patches"]),
                                "posture_filters": ",".join(cam["posture_filters"])})
            conversation = [{"role": "patient", "text": sc["patient_text"]}]
            for r in range(runs):
                seed = settings["seed0"] + r
                t0 = time.time()
                error = None
                try:
                    verdict = llm_call(conversation, signs, skin_status, "NORMAL_COMPLETE", sc.get("language", "en"),
                                       temperature=settings["temperature"], seed=seed, timeout=args.timeout)
                except Exception as e:  # recorded, never silently dropped
                    verdict, error = {}, f"{type(e).__name__}: {e}"
                wall = round(time.time() - t0, 3)
                audit_mode = str(verdict.get("audit_mode", ""))
                rec = {
                    "id": sc["id"], "run": r, "seed": seed, "image_file": sc["image_file"], "image_file_skin": skin_file,
                    "language": sc.get("language"), "visual_signs_sent": signs, "skin_status_sent": skin_status,
                    "esi_pred": valid_esi(verdict.get("esi_level")), "esi_raw": verdict.get("esi_level"),
                    "urgency": verdict.get("urgency"), "audit_mode": audit_mode,
                    "fallback": audit_mode.startswith("DETERMINISTIC_FALLBACK"),
                    "llm_latency_s": verdict.get("inference_latency_sec"), "wall_time_s": wall,
                    "exceeds_kiosk_timeout": (verdict.get("inference_latency_sec") or wall) > KIOSK_TIMEOUT_S,
                    "scenario_type": verdict.get("scenario_type"), "error": error,
                    "soap": {k: verdict.get(k) for k in ("subjective", "objective", "assessment", "plan")},
                }
                jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jl.flush()
            print(f"[{n}/{len(scenarios)}] {sc['id']}: camera={cam['posture_label']}/{cam['skin_status']} "
                  f"-> ESI {rec['esi_pred']} ({wall:.1f}s{', FALLBACK' if rec['fallback'] else ''})")
    write_csv(out / "vision.csv", vision_rows, list(vision_rows[0].keys()) if vision_rows else ["id"])
    print(f"\nSaved {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("A", "B"), required=True)
    ap.add_argument("--pipeline", choices=("fixed", "original"), default="fixed")
    ap.add_argument("--runs", type=int, default=None, help="override runs per scenario (default A=1, B=10)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS_CSV)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenarios (smoke test)")
    ap.add_argument("--no-posture", action="store_true", help="(future M0/M2) send no posture signs")
    ap.add_argument("--no-skin", action="store_true", help="(future M0/M1) send no skin result")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
