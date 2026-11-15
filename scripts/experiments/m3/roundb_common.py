"""Shared pieces of the round-B scripts: paths, a disk cache around the LLM call, scoring helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

from m3_common import DOCS_DIR, PROJECT_ROOT, RESULTS_DIR

ROUNDB_DIR = RESULTS_DIR / "roundB"
SELECTION_JSON = ROUNDB_DIR / "prompt_selection.json"
DEV_SCENARIOS_CSV = DOCS_DIR / "dev_scenarios.csv"
RUN_DIR = ROUNDB_DIR / "run_main"
CANDIDATES = [("v1", False), ("v2", False), ("v2", True), ("v3", False), ("v3", True)]   # simplest first
INFRASTRUCTURE_FAILURES = ("TIMEOUT", "UNREACHABLE", "OLLAMA_HTTP")


def candidate_name(prompt_version: str, from_steps: bool) -> str:
    return prompt_version if prompt_version == "v1" else f"{prompt_version}-{'steps' if from_steps else 'level'}"


def intake_from_text(text: str) -> Optional[dict]:
    """What the kiosk's structured intake would hold for a scripted patient text: an explicit
    pain score ("8/10", "แปดเต็มสิบ", "six out of ten") and the onset words, if present."""
    from triage_kiosk.triage.intake import extract_onset, extract_severity
    if not text.strip():
        return None
    score, label = extract_severity(text, asked=False)
    return {"complaint": text, "pain_score_0_10": score, "severity": label, "onset": extract_onset(text),
            "not_answered": []}


def signs_from_label(posture: str) -> dict:
    posture = (posture or "none").strip().lower()
    return {"chest_clutching": posture == "chest", "abdominal_clutching": posture == "abdomen",
            "head_clutching": posture == "head"}


class LLMCache:
    """Disk cache of LLM answers, keyed by model + prompt + options. Makes every run resumable:
    a repeated call (same prompt, temperature and seed) is read back instead of sent again.
    Infrastructure failures (timeout, Ollama not reachable) are never stored."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.hits = self.misses = 0
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec

    @staticmethod
    def key(model: str, prompt: str, temperature: float, seed, num_predict: int) -> str:
        blob = json.dumps([model, prompt, float(temperature), seed, int(num_predict)], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def install(self, engine, model: str) -> None:
        """Wrap ``engine._call_llm`` with the cache."""
        real = engine._call_llm
        cache = self
        cache.path.parent.mkdir(parents=True, exist_ok=True)

        def cached(prompt, temperature, seed, timeout, num_predict):
            k = cache.key(model, prompt, temperature, seed, num_predict)
            if k in cache.data:
                cache.hits += 1
                rec = cache.data[k]
                engine.last_call = {"cached": True, "latency": rec["latency"]}
                return rec["out"], rec["fallback_reason"], rec["latency"]
            cache.misses += 1
            out, reason, latency = real(prompt, temperature, seed, timeout, num_predict)
            engine.last_call = {"cached": False, "latency": latency}
            if not (reason and any(x in reason for x in INFRASTRUCTURE_FAILURES)):
                rec = {"key": k, "out": out, "fallback_reason": reason, "latency": latency}
                cache.data[k] = rec
                with open(cache.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return out, reason, latency

        engine._call_llm = cached
        engine.last_call = {}


def make_engine(cache_path: Path, llm=None):
    """ReasoningEngine with the cached LLM call. ``llm`` replaces the real call in tests."""
    from triage_kiosk.config import SystemConfig
    from triage_kiosk.triage.reasoning import ReasoningEngine
    engine = ReasoningEngine.__new__(ReasoningEngine)
    engine.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"
    if llm is not None:
        engine._call_llm = llm
    cache = LLMCache(cache_path)
    cache.install(engine, SystemConfig.OLLAMA_MODEL)
    return engine, cache


def ollama_ready() -> tuple[bool, str]:
    import requests
    from triage_kiosk.config import SystemConfig
    try:
        tags = requests.get(f"{SystemConfig.OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])
    except Exception as e:  # noqa: BLE001
        return False, f"Ollama is not reachable at {SystemConfig.OLLAMA_HOST} ({type(e).__name__}). Open the Ollama app first."
    names = [t.get("name") for t in tags]
    if SystemConfig.OLLAMA_MODEL not in names:
        return False, f"Model {SystemConfig.OLLAMA_MODEL} is not pulled (have: {names}). Run: ollama pull llama3.2"
    info = next(t for t in tags if t.get("name") == SystemConfig.OLLAMA_MODEL)
    return True, json.dumps({"name": info.get("name"), "digest": info.get("digest"),
                             "details": info.get("details")}, ensure_ascii=False)


def final_level(base: Optional[int], caps: list[dict], visual_band: Optional[str] = None,
                policy: Optional[str] = None) -> Optional[int]:
    """Final ESI = the base level limited by the recorded safety-floor caps. With ``policy`` the
    visual-band cap is recomputed for another floor policy (sensitivity analysis)."""
    from triage_kiosk.triage import fusion
    if base is None:
        return None
    limits = [c["cap"] for c in caps if policy is None or c["source"] != "visual_band"]
    if policy is not None and visual_band:
        cap = fusion.esi_cap(visual_band, policy)
        if cap is not None:
            limits.append(cap)
    return min([base] + limits)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f) if r.get("id")]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
