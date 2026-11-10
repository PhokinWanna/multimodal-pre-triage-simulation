"""Send realistic fake kiosk cases to the nurse dashboard (no camera or microphone needed).

Usage (with the dashboard running):
    python scripts/dashboard/send_demo_cases.py
    python scripts/dashboard/send_demo_cases.py --url http://127.0.0.1:8000/api/triage/submit_case

All data is invented for testing and demonstration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_URL = "http://127.0.0.1:8000/api/triage/submit_case"


def _case(n: int, lang: str, exit_reason: str, turns: list[tuple[str, str]], signs: dict, skin: str,
          verdict: dict | None) -> dict:
    return {
        "case_id": f"CASE-DEMO-{int(time.time())}-{n}",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": lang,
        "pdpa_consented": True,
        "exit_reason": exit_reason,
        "conversation": [{"role": r, "text": t} for r, t in turns],
        "cumulative_visual_signs": {"chest_clutching": False, "abdominal_clutching": False,
                                    "head_clutching": False, **signs},
        "facial_skin_status": skin,
        "visual_evidence_path": None,
        "triage_verdict": verdict,
    }


def _verdict(esi: int, urgency: str, scenario: str, s: str, o: str, a: str, p: str,
             mode: str = "LLM_INFERENCE_VALIDATED") -> dict:
    return {"esi_level": esi, "urgency": urgency, "scenario_type": scenario,
            "subjective": s, "objective": o, "assessment": a, "plan": p, "audit_mode": mode}


def demo_payloads() -> list[dict]:
    return [
        _case(1, "th", "EMERGENCY_INTERRUPT",
              [("pdpa_response", "ยินยอมครับ"), ("kiosk_proactive", "วันนี้มีอาการอะไรครับ"),
               ("patient", "เจ็บแน่นหน้าอกมาก ร้าวไปแขนซ้าย เหงื่อออกครับ")],
              {"chest_clutching": True}, "RED_FLUSHING",
              _verdict(2, "Emergent", "SCENARIO_A_CONFIRMED", "Severe chest tightness radiating to left arm, sweating.",
                       "Chest clutching sustained; facial flushing.", "Possible acute coronary syndrome.",
                       "Immediate ECG and physician review.")),
        _case(2, "en", "NORMAL_COMPLETE",
              [("pdpa_response", "Yes, I agree"), ("kiosk_proactive", "What brings you in today?"),
               ("patient", "Stomach pain since this morning, about 6 out of 10"),
               ("kiosk_agent", "Where is the pain?"), ("patient", "Lower right side")],
              {"abdominal_clutching": True}, "NORMAL",
              _verdict(3, "Urgent", "SCENARIO_A_CONFIRMED", "Right lower abdominal pain since morning, 6/10.",
                       "Abdominal guarding observed.", "Rule out appendicitis; likely needs labs and imaging.",
                       "Labs, abdominal ultrasound, physician assessment.")),
        _case(3, "th", "NORMAL_COMPLETE",
              [("pdpa_response", "ยินยอมค่ะ"), ("kiosk_proactive", "วันนี้มีอาการอะไรคะ"),
               ("patient", "ปวดหัวมาสองวัน กินยาแล้วไม่หายค่ะ")],
              {"head_clutching": True}, "NORMAL",
              _verdict(4, "Less Urgent", "SCENARIO_A_CONFIRMED", "Headache for two days, not relieved by medication.",
                       "Head holding observed.", "Tension-type headache likely; no red flags reported.",
                       "One medication; reassess if worse.")),
        _case(4, "en", "NORMAL_COMPLETE",
              [("pdpa_response", "Yes"), ("kiosk_proactive", "What brings you in today?"),
               ("patient", "I need a refill of my blood pressure medicine")],
              {}, "NORMAL",
              _verdict(5, "Non-urgent", "SCENARIO_B_SUBJECTIVE_ONLY", "Requests medication refill.",
                       "No distress observed.", "Non-urgent medication refill.", "Route to OPD.",
                       mode="DETERMINISTIC_FALLBACK")),
        _case(5, "th", "USER_CANCELLED",
              [("pdpa_response", "ยินยอมครับ"), ("kiosk_proactive", "วันนี้มีอาการอะไรครับ"),
               ("patient", "ขอยกเลิกครับ")],
              {}, "NORMAL", None),
    ]


def post(url: str, payload: dict, key: str = "") -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Kiosk-Key"] = key
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between cases (to watch them arrive)")
    parser.add_argument("--key", default=os.getenv("NURSE_DASHBOARD_KEY") or os.getenv("DASHBOARD_KIOSK_KEY", ""),
                        help="kiosk key (default: $NURSE_DASHBOARD_KEY or $DASHBOARD_KIOSK_KEY)")
    args = parser.parse_args()

    try:
        for payload in demo_payloads():
            status, body = post(args.url, payload, args.key)
            print(f"{payload['case_id']}: {status} {body}")
            if status == 401:
                raise SystemExit("Rejected (401): set --key or NURSE_DASHBOARD_KEY to the dashboard's DASHBOARD_KIOSK_KEY.")
            time.sleep(args.delay)
        event = {"event_type": "nurse_assistance_requested", "reason": "consent_refused", "event_id": uuid.uuid4().hex}
        status, body = post(args.url, event, args.key)
        print(f"Consent-refusal alert: {status} {body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach the dashboard at {args.url} ({e.reason}). Is it running?")


if __name__ == "__main__":
    main()
