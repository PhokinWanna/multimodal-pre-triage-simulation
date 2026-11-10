"""Hardware/software pre-flight check for the kiosk Mac. Read-only: sends no case to the dashboard.

Run from the repository root in the kiosk virtual environment:
    PYTHONPATH=src python scripts/checks/kiosk_preflight.py            # quick checks
    PYTHONPATH=src python scripts/checks/kiosk_preflight.py --full     # + mic recording, speaker, Whisper, LLM timing

Prints one line per check: OK / WARN / FAIL.
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from triage_kiosk.config import SystemConfig

RESULTS = []


def report(status, name, detail=""):
    RESULTS.append(status)
    print(f"[{status:4}] {name}" + (f" -> {detail}" if detail else ""))


def check_packages():
    for mod in ["numpy", "cv2", "mediapipe", "requests", "speech_recognition", "pyaudio",
                "faster_whisper", "gtts", "ai_edge_litert"]:
        try:
            m = importlib.import_module(mod)
            report("OK", f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            hint = " (macOS: brew install portaudio, then pip install PyAudio)" if mod == "pyaudio" else ""
            report("FAIL", f"import {mod}", f"{e}{hint}")


def check_files():
    missing = []
    for lang, table in SystemConfig.TEMPLATES.items():
        for key, name in table.items():
            if not os.path.isfile(os.path.join(SystemConfig.AUDIO_TEMPLATE_DIR, name)):
                missing.append(f"{lang}/{key}")
    report("OK" if not missing else "FAIL", "audio templates (th + en)", ", ".join(missing) or "all present")
    model = SystemConfig.TFLITE_SKIN_MODEL_PATH
    meta = os.path.splitext(model)[0] + ".json"
    report("OK" if os.path.isfile(model) else "FAIL", "skin CNN model", model)
    report("OK" if os.path.isfile(meta) else "WARN", "skin CNN metadata (threshold)", meta)
    try:
        os.makedirs(SystemConfig.OUTBOX_SPOOL_DIR, exist_ok=True)
        left = [n for n in os.listdir(SystemConfig.OUTBOX_SPOOL_DIR) if n.endswith(".json")]
        report("OK" if not left else "WARN", "outbox spool folder writable",
               f"{len(left)} undelivered record(s) waiting" if left else "empty")
    except OSError as e:
        report("FAIL", "outbox spool folder", str(e))
    if platform.system() == "Darwin":
        report("OK" if shutil.which("afplay") else "FAIL", "afplay (speaker playback)")


def check_camera():
    import cv2
    cap = cv2.VideoCapture(SystemConfig.CAMERA_INDEX)
    ok, frame = (cap.read() if cap.isOpened() else (False, None))
    cap.release()
    if ok and frame is not None:
        report("OK", f"camera index {SystemConfig.CAMERA_INDEX}", f"frame {frame.shape[1]}x{frame.shape[0]}")
    else:
        report("FAIL", f"camera index {SystemConfig.CAMERA_INDEX}",
               "cannot read a frame (macOS: System Settings > Privacy > Camera for Terminal/IDE)")


def check_microphone(record):
    try:
        import speech_recognition as sr
        names = sr.Microphone.list_microphone_names()
        report("OK" if names else "FAIL", "microphones found", "; ".join(names[:4]))
        if not record:
            return None
        r = sr.Recognizer()
        print("        Speak a short sentence now (4 s)...")
        with sr.Microphone() as src:
            audio = r.record(src, duration=4)
        import numpy as np
        pcm = np.frombuffer(audio.get_raw_data(convert_rate=16000, convert_width=2), dtype=np.int16)
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        report("OK" if rms > 150 else "WARN", "microphone recording level",
               f"RMS {rms:.0f} (very low = muted / no permission: Privacy > Microphone)")
        return pcm
    except Exception as e:  # noqa: BLE001
        report("FAIL", "microphone", str(e))
        return None


def check_voices(speak):
    if platform.system() != "Darwin":
        report("WARN", "offline voices", "macOS only; elsewhere the generic recordings are used")
        return
    from triage_kiosk.audio.engine import pick_voices
    listing = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True).stdout
    voices = pick_voices(listing)
    for lang in ("th", "en"):
        v = voices.get(lang)
        hint = ("System Settings > Accessibility > Spoken Content > System voice > Manage Voices > "
                "Thai > Kanya" if lang == "th" else "install any English voice")
        report("OK" if v else "WARN", f"offline {lang} voice for follow-up questions",
               v or f"missing -> kiosk will play the generic question instead ({hint})")
        if v and speak:
            text = "เป็นมานานกี่ชั่วโมงแล้วคะ" if lang == "th" else "How long have you had this pain?"
            subprocess.call(["/usr/bin/say", "-v", v, text])


def check_speaker():
    path = os.path.join(SystemConfig.AUDIO_TEMPLATE_DIR, SystemConfig.TEMPLATES["th"]["greeting"])
    player = ["afplay", path] if platform.system() == "Darwin" else None
    if not player:
        report("WARN", "speaker", "non-macOS: test pygame playback manually")
        return
    t0 = time.time()
    rc = subprocess.call(player)
    report("OK" if rc == 0 else "FAIL", "speaker plays Thai greeting", f"{time.time() - t0:.1f} s - did you hear it?")


def check_whisper(pcm):
    try:
        from faster_whisper import WhisperModel
        t0 = time.time()
        m = WhisperModel(SystemConfig.FASTER_WHISPER_MODEL, device=SystemConfig.FASTER_WHISPER_DEVICE,
                         compute_type=SystemConfig.FASTER_WHISPER_COMPUTE,
                         cpu_threads=SystemConfig.FASTER_WHISPER_CPU_THREADS)
        report("OK", f"Faster-Whisper '{SystemConfig.FASTER_WHISPER_MODEL}' loads", f"{time.time() - t0:.1f} s")
        if pcm is not None:
            t0 = time.time()
            segs, info = m.transcribe(pcm.astype("float32") / 32768.0, beam_size=2, vad_filter=True)
            text = " ".join(s.text for s in segs).strip()
            report("OK" if text else "WARN", "Whisper transcription",
                   f"'{text}' (lang={info.language}, {time.time() - t0:.1f} s)")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "Faster-Whisper", f"{e} (first run downloads the model: needs internet once)")


def check_ollama(timing):
    import requests
    host = SystemConfig.OLLAMA_HOST
    try:
        tags = requests.get(f"{host}/api/tags", timeout=3).json().get("models", [])
        names = [t["name"] for t in tags]
        has = SystemConfig.OLLAMA_MODEL in names
        report("OK" if has else "FAIL", f"Ollama model '{SystemConfig.OLLAMA_MODEL}'",
               "pulled" if has else f"not pulled; run: ollama pull {SystemConfig.OLLAMA_MODEL} (have: {names})")
        if SystemConfig.OLLAMA_TURN_MODEL != SystemConfig.OLLAMA_MODEL:
            ok = SystemConfig.OLLAMA_TURN_MODEL in names
            report("OK" if ok else "FAIL", f"Ollama turn model '{SystemConfig.OLLAMA_TURN_MODEL}'",
                   "pulled" if ok else f"run: ollama pull {SystemConfig.OLLAMA_TURN_MODEL}")
        if not has or not timing:
            return
    except Exception as e:  # noqa: BLE001
        report("FAIL", "Ollama server", f"{host} unreachable: {e} (start the Ollama app / `ollama serve`)")
        return
    for label, n, limit, model in [
            ("interview turn", 85, SystemConfig.OLLAMA_TURN_TIMEOUT, SystemConfig.OLLAMA_TURN_MODEL),
            ("final verdict", 256, SystemConfig.OLLAMA_FINAL_TIMEOUT, SystemConfig.OLLAMA_MODEL)]:
        payload = {"model": model, "stream": False, "format": "json", "keep_alive": -1,
                   "prompt": "Return JSON {\"esi_level\":3,\"assessment\":\"<two sentences about abdominal pain>\"}",
                   "options": {"temperature": 0, "num_predict": n}}
        times = []
        for _ in range(3):
            t0 = time.time()
            requests.post(f"{host}/api/generate", json=payload, timeout=60)
            times.append(time.time() - t0)
        worst = max(times[1:])  # first call may include model loading
        status = "OK" if worst < limit else "FAIL"
        report(status, f"LLM latency, {label}",
               f"{', '.join(f'{t:.1f}' for t in times)} s vs timeout {limit:.0f} s"
               + ("" if status == "OK" else " -> real turns would fall back to the rule-based path"))


def check_dashboard():
    import requests
    ep = SystemConfig.NURSE_DASHBOARD_ENDPOINT
    root = "{0.scheme}://{0.netloc}".format(urlsplit(ep))
    try:
        r = requests.get(f"{root}/api/health", timeout=3)
        report("OK" if r.status_code == 200 else "FAIL", "nurse dashboard reachable", f"{root} -> {r.status_code}")
    except Exception as e:  # noqa: BLE001
        report("WARN", "nurse dashboard reachable",
               f"{root} unreachable ({type(e).__name__}); cases will wait in the outbox until it is up")
    report("OK" if SystemConfig.NURSE_DASHBOARD_KEY else "WARN", "NURSE_DASHBOARD_KEY set",
           "" if SystemConfig.NURSE_DASHBOARD_KEY else "empty: dashboard will answer 401 if it requires a kiosk key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="record mic, play speaker, load Whisper, time the LLM")
    a = ap.parse_args()
    print(f"Python {sys.version.split()[0]} on {platform.platform()}\n")
    check_packages()
    check_files()
    check_camera()
    pcm = check_microphone(record=a.full)
    check_voices(speak=a.full)
    if a.full:
        check_speaker()
        check_whisper(pcm)
    check_ollama(timing=a.full)
    check_dashboard()
    fails = RESULTS.count("FAIL")
    print(f"\n{RESULTS.count('OK')} OK, {RESULTS.count('WARN')} WARN, {fails} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
