"""Simulated end-to-end check of the kiosk state machine (no camera, mic, speaker or Ollama).

Runs the REAL ``SmartReceptionTriageKiosk._fsm_worker`` and the REAL ``ReasoningEngine``
logic, with fakes for audio (scripted patient replies), the camera telemetry, the
Ollama HTTP server and the dashboard outbox. Each scenario states the SAFE expected
behaviour and prints PASS/FAIL, so it can be re-run after fixes.

Run from the repository root:
    PYTHONPATH=src python scripts/checks/kiosk_fsm_sim.py
"""

import copy
import json
import logging
import sys
import threading
import time as _time
import types

# Hardware-only audio libraries are not needed for the simulation.
for _name in ("speech_recognition", "gtts", "pygame", "faster_whisper"):  # noqa
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.gTTS = object
        _m.mixer = types.SimpleNamespace(init=lambda: None, quit=lambda: None)
        sys.modules[_name] = _m

import requests  # noqa: E402

from triage_kiosk import main as kiosk_main  # noqa: E402
from triage_kiosk.audio.engine import AudioEngine  # noqa: E402
from triage_kiosk.config import KioskState, SystemConfig  # noqa: E402
from triage_kiosk.triage import reasoning as reasoning_mod  # noqa: E402

# Speed up the FSM's own sleeps.
kiosk_main.time = types.SimpleNamespace(time=_time.time, strftime=_time.strftime,
                                        sleep=lambda s: _time.sleep(min(s, 0.005)))

EVENTS = []          # ordered log of everything the "patient" would see / hear
FSM_ERRORS = []


class _ErrHandler(logging.Handler):
    def emit(self, record):
        if "FSM Encounter Exception" in record.getMessage():
            FSM_ERRORS.append(record.getMessage())


_fsm_log = logging.getLogger("triage_kiosk.main")
_fsm_log.addHandler(_ErrHandler())
_fsm_log.setLevel(logging.ERROR)
_fsm_log.propagate = False
logging.getLogger().setLevel(logging.CRITICAL)


class FakeAudio:
    def __init__(self, replies, mic_error=False, voice=True, fail_after=None):
        self.replies = list(replies)
        self.mic_error = mic_error
        self.voice = voice
        self.fail_after = fail_after
        self.n_listen = 0

    def listen(self, language="th"):
        self.n_listen += 1
        if self.mic_error or (self.fail_after is not None and self.n_listen > self.fail_after):
            raise OSError("Invalid input device (simulated unplugged microphone)")
        r = self.replies.pop(0) if self.replies else ""
        EVENTS.append(("HEARD", r))
        # like Whisper's Thai/English choice: Latin text -> English, Thai text -> Thai
        self.last_language = ((None, 0.0) if not r else ("en", 0.95) if r.isascii() else ("th", 0.95))
        return r

    def play_template(self, key, lang="th", block=True):
        EVENTS.append(("PLAY", key))

    def calibrate(self, duration=None):
        pass

    def can_speak(self, lang):
        return self.voice

    def speak_text(self, text, lang="th", block=True):
        if self.voice and text:
            EVENTS.append(("SAY", text))
            return True
        return False

    def speak_fixed(self, text, lang="th", block=True):
        if self.speak_text(text, lang, block):
            return "say"
        EVENTS.append(("PLAY", "inquiry"))
        return "template"

    def speak_off_topic(self, lang="th", block=True):
        EVENTS.append(("PLAY", "off_topic_tts"))

    detect_wake_word = staticmethod(AudioEngine.detect_wake_word)


class FakeOutbox:
    def __init__(self):
        self.cases, self.handoffs = [], []

    def enqueue_case(self, case):
        EVENTS.append(("DISPATCH", case.get("exit_reason")))
        self.cases.append(copy.deepcopy(case))

    def enqueue_nurse_assistance(self, event):
        EVENTS.append(("HANDOFF", event.get("reason")))
        self.handoffs.append(dict(event))


def fake_ollama(mode, turn_reply=None, final_reply=None):
    """mode: ok | timeout | 404."""
    def post(url, json=None, timeout=None, **kw):
        final = json["options"].get("num_predict", 0) >= 150
        EVENTS.append(("LLM_CALL", "final" if final else "turn"))
        if mode == "timeout":
            raise requests.Timeout("simulated: M1 needs ~7.3 s, timeout is %.1f s" % timeout)
        resp = types.SimpleNamespace(status_code=200 if mode == "ok" else 404)
        body = final_reply if final else turn_reply
        resp.json = lambda: {"response": _json.dumps(body or {})}
        return resp
    return post


_json = json


def run(replies, telemetry=None, ollama="ok", turn_reply=None, final_reply=None,
        mic_error=False, max_seconds=3.0, voice=True, fail_after=None, mode="structured"):
    SystemConfig.INTERVIEW_MODE = mode
    EVENTS.clear()
    FSM_ERRORS.clear()
    reasoning_mod.requests.post = fake_ollama(ollama, turn_reply, final_reply)

    k = kiosk_main.SmartReceptionTriageKiosk.__new__(kiosk_main.SmartReceptionTriageKiosk)
    k.audio = FakeAudio(replies, mic_error=mic_error, voice=voice, fail_after=fail_after)
    r = reasoning_mod.ReasoningEngine.__new__(reasoning_mod.ReasoningEngine)
    r.endpoint = "http://fake-ollama/api/generate"
    k.reasoning = r
    k.conveyor_belt = FakeOutbox()
    k.evidence_dir = "/tmp"
    k.last_heartbeat = _time.time()
    k.current_state = KioskState.PREPARE_STANDBY
    k.active_language = "th"
    k.running = True
    k.current_strikes = 0
    k.telemetry_lock = threading.Lock()
    base = {"patient_present": True, "sustained_gestures": {}, "skin_status": "NORMAL",
            "dominant_current_gesture": "NONE"}
    base.update(telemetry or {})
    k.latest_telemetry = base
    k.case_container = {}
    k.last_fault_report = 0.0
    k.refusal_handoff = None
    k._reset_case_container()

    t = threading.Thread(target=k._fsm_worker, daemon=True)
    t.start()
    end = _time.time() + max_seconds
    while _time.time() < end:
        if k.conveyor_belt.cases or (k.conveyor_belt.handoffs and k.current_state == KioskState.PREPARE_STANDBY):
            _time.sleep(0.05)
            break
        _time.sleep(0.01)
    k.running = False
    t.join(1.0)
    case = k.conveyor_belt.cases[0] if k.conveyor_belt.cases else None
    return case, k.conveyor_belt.handoffs, list(EVENTS), list(FSM_ERRORS)


RESULTS = []


def check(sid, title, ok, detail):
    RESULTS.append((sid, "PASS" if ok else "FAIL", title, detail))


GREET = ["สวัสดีค่ะ", "ภาษาไทยค่ะ"]               # standby wake word, language choice
GREET_EN = ["hello", "English please"]


def plays(events, key):
    return sum(1 for e in events if e == ("PLAY", key))


# ---------------------------------------------------------------- consent
case, hand, ev, err = run(GREET + ["", "", "", ""])
check("C1", "PDPA: silence must NOT count as consent",
      not (case and case.get("pdpa_consented")),
      f"case dispatched={case is not None}, pdpa_consented={case and case.get('pdpa_consented')}")

case, hand, ev, err = run(GREET + ["ไม่มีปัญหาค่ะ ยินยอมค่ะ", "", "", ""])
check("C2", "PDPA: 'ไม่มีปัญหาค่ะ ยินยอมค่ะ' (no problem, I consent) is consent",
      bool(case and case.get("pdpa_consented")), f"refusal handoffs={len(hand)}")

case, hand, ev, err = run(GREET_EN + ["Yes I agree, I know the policy", "", "", ""])
check("C3", "PDPA: 'Yes I agree, I know the policy' is consent",
      bool(case and case.get("pdpa_consented")), f"refusal handoffs={len(hand)}")

case, hand, ev, err = run(GREET + ["อะไรนะคะ", ""])
check("C5", "PDPA: unclear twice -> re-asked once, then nurse handoff 'consent_unclear', no case data",
      case is None and len(hand) == 1 and hand[0]["reason"] == "consent_unclear"
      and any(e[0] == "SAY" for e in ev), f"case={case is not None}, handoffs={hand}")

case, hand, ev, err = run(GREET + ["ไม่ยินยอมค่ะ"])
check("C4", "PDPA: 'ไม่ยินยอมค่ะ' (I do not consent) -> no case data, anonymous handoff only",
      case is None and len(hand) == 1 and set(hand[0]) <= {"event_type", "reason", "event_id"},
      f"case={case is not None}, handoffs={hand}")

# ---------------------------------------------------------------- normal interview
turn = {"action": "CONTINUE_TRIAGE", "target_language": "th", "is_off_topic": False,
        "spoken_response": "เริ่มปวดตั้งแต่เมื่อไหร่คะ"}
final = {"esi_level": 3, "urgency": "Urgent", "subjective": "s", "objective": "o",
         "assessment": "a", "plan": "p", "audit_mode": "LLM_INFERENCE_VALIDATED"}
case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดท้องค่ะ", "เป็นตั้งแต่เมื่อวาน ปวด 7 เต็ม 10", ""],
                          turn_reply=turn, final_reply=final, mode="llm")
check("N1", "Normal Thai interview completes and is dispatched with an ESI",
      bool(case and case["triage_verdict"] and case["triage_verdict"].get("esi_level") == 3),
      f"exit={case and case['exit_reason']}, verdict ESI={case and (case['triage_verdict'] or {}).get('esi_level')}")
def heard_texts(events, lang="th"):
    out = {e[1] for e in events if e[0] == "SAY"}
    out |= {SystemConfig.SCRIPT_TEXTS[lang].get(e[1], "") for e in events if e[0] == "PLAY"}
    return out


def unheard(case, events, lang="th"):
    kiosk = [t["text"] for t in (case or {}).get("conversation", []) if t["role"].startswith("kiosk")]
    return [t for t in kiosk if t not in heard_texts(events, lang)]


check("N2", "[llm mode] Every kiosk sentence in the transcript was really spoken; the LLM question is voiced",
      bool(case) and not unheard(case, ev) and ("SAY", "เริ่มปวดตั้งแต่เมื่อไหร่คะ") in ev,
      f"logged-but-not-heard={unheard(case, ev)}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดท้องค่ะ", "เป็นตั้งแต่เมื่อวาน ปวด 7 เต็ม 10", ""],
                          turn_reply=turn, final_reply=final, voice=False, mode="llm")
check("N2b", "[llm mode] No offline voice installed -> generic recording played AND logged (no fake transcript)",
      bool(case) and not unheard(case, ev), f"logged-but-not-heard={unheard(case, ev)}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดท้องค่ะ", "เป็นตั้งแต่เมื่อวาน ปวด 7 เต็ม 10", ""],
                          turn_reply=dict(turn, action="CANCEL_AND_EXIT"), final_reply=final, mode="llm")
check("N4", "[llm mode] LLM alone cannot cancel the case (patient never asked to cancel)",
      bool(case and case["exit_reason"] != "USER_CANCELLED"), f"exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "", "", ""])
check("N3", "Consent then total silence -> encounter closes, kiosk resets",
      case is not None and not err, f"exit={case and case['exit_reason']}, errors={len(err)}")

# ---------------------------------------------------------------- emergency
case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "แน่นหน้าอกมาก หายใจลำบากค่ะ"], ollama="timeout")
esi = case and (case["triage_verdict"] or {}).get("esi_level")
check("E1", "Emergency words ('แน่นหน้าอกมาก') + LLM timeout -> ESI 1-2",
      esi in (1, 2), f"exit={case and case['exit_reason']}, final ESI={esi}, "
      f"urgency={case and case['triage_verdict'].get('urgency')}")
order = [e for e in ev if e[0] in ("PLAY", "LLM_CALL") and e[1] in ("emergency_alert", "final")]
check("E2", "Emergency alert is played BEFORE the slow final LLM synthesis",
      bool(order) and order[0] == ("PLAY", "emergency_alert"), f"order={order}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "เจ็บมากค่ะ"],
                          telemetry={"dominant_current_gesture": "CHEST_PAIN"},
                          ollama="ok", final_reply=dict(final, esi_level=2), mode="llm")
check("E3", "[llm mode] Chest clutching + 'เจ็บมาก' -> emergency alert played exactly once",
      plays(ev, "emergency_alert") == 1, f"emergency_alert played {plays(ev, 'emergency_alert')}x")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "เจ็บหน้าอกค่ะ เป็นมา 2 ชั่วโมง", "ไม่มากค่ะ", ""],
                          telemetry={"sustained_gestures": {"chest_clutching": True}},
                          turn_reply=turn, final_reply=dict(final, esi_level=4))
esi = case and case["triage_verdict"].get("esi_level")
check("E4", "LLM says ESI 4 although sustained chest clutching was seen -> safety floor ESI <= 2",
      esi in (1, 2), f"final ESI={esi}")

# ---------------------------------------------------------------- LLM failures
case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัวค่ะ", "เมื่อเช้า ปวดมาก", ""], ollama="404")
check("L1", "Ollama answers 404 (model not pulled) -> fallback verdict, case dispatched",
      case is not None and not err,
      f"case dispatched={case is not None}, FSM exceptions={len(err)} first={err[:1]}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัวค่ะ", "เมื่อเช้า ปวดมาก", ""],
                          turn_reply=turn, final_reply={"urgency": "Urgent", "esi_level": 0})
esi = case and case["triage_verdict"].get("esi_level")
check("L2", "LLM returns an invalid verdict (esi_level 0) -> rejected / fallback used",
      esi in (1, 2, 3, 4, 5), f"final ESI={esi}, verdict keys={case and sorted(case['triage_verdict'])}")

# ---------------------------------------------------------------- keyword false triggers
case, hand, ev, err = run(GREET_EN + ["Yes I agree", "I can't stop coughing since yesterday", "", ""],
                          turn_reply=dict(turn, target_language="en", spoken_response="How severe?"),
                          final_reply=final, mode="llm")
check("K1", "[llm mode] 'I can't stop coughing' must NOT cancel the case",
      bool(case and case["exit_reason"] != "USER_CANCELLED"), f"exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ยาความดันหมดแล้วค่ะ เวียนหัว", "", ""],
                          turn_reply=turn, final_reply=final, mode="llm")
n_pat = sum(1 for t in (case or {}).get("conversation", []) if t["role"] == "patient")
check("K2", "[llm mode] 'ยาความดันหมดแล้ว' (my BP medicine ran out) must NOT end the interview at turn 1",
      bool(case and n_pat > 1) or bool(case and any(e == ("LLM_CALL", "turn") for e in ev)),
      f"exit={case and case['exit_reason']}, patient turns={n_pat}")

case, hand, ev, err = run(["this is the waiting room right?", ""], max_seconds=0.5)
greet_lang = [e for e in ev if e == ("PLAY", "greeting")]
check("K3", "'this is...' contains 'hi' but is not a wake word (no false English switch)",
      not AudioEngine.detect_wake_word("this is the waiting room right?")[0],
      f"wake word detected={AudioEngine.detect_wake_word('this is the waiting room right?')}; "
      "note: by design ANY speech from a person standing in the camera zone starts an encounter")


# ---------------------------------------------------------------- structured interview (default mode)
from triage_kiosk.triage.intake import QUESTIONS, REPROMPTS  # noqa: E402


def says(events, text):
    return sum(1 for e in events if e == ("SAY", text))


case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัวค่ะ", "7", "ตั้งแต่เมื่อเช้า", "ไม่มีแล้วค่ะ"],
                          final_reply=final)
it = (case or {}).get("intake") or {}
check("S1", "Structured: complaint -> 0-10 score -> onset, each asked ONCE, then 'anything else'",
      bool(case) and says(ev, QUESTIONS["th"]["severity"]) == 1 and says(ev, QUESTIONS["th"]["onset"]) == 1
      and plays(ev, "anything_else") == 1 and not any(e == ("LLM_CALL", "turn") for e in ev)
      and it.get("pain_score_0_10") == 7 and it.get("onset") and not unheard(case, ev),
      f"intake={it}, exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัว 8 เต็ม 10 ตั้งแต่เมื่อวานค่ะ", ""],
                          final_reply=dict(final, esi_level=4))
check("S2", "Structured: score + onset in the first answer -> no extra questions; pain 8/10 -> ESI <= 3",
      bool(case) and says(ev, QUESTIONS["th"]["severity"]) == 0 and says(ev, QUESTIONS["th"]["onset"]) == 0
      and case["triage_verdict"]["esi_level"] <= 3,
      f"intake={case and case.get('intake')}, ESI={case and case['triage_verdict']['esi_level']}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดท้องค่ะ", "อืม", "ไม่รู้สิ", "ไม่แน่ใจค่ะ", ""],
                          final_reply=final)
check("S3", "Structured: unclear score -> one clearer re-ask, then move on (no loop); 'not sure' onset accepted",
      bool(case) and says(ev, QUESTIONS["th"]["severity"]) == 1 and says(ev, REPROMPTS["th"]["severity"]) == 1
      and case["exit_reason"] == "NORMAL_COMPLETE" and (case.get("intake") or {}).get("onset") == "unknown",
      f"exit={case and case['exit_reason']}, intake={case and case.get('intake')}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดท้องค่ะ", "blue", "banana", "potato"], final_reply=final)
check("S4", "Structured: 3 answers in a row that do not answer -> possible confusion (AMS), nurse alert, ESI <= 2",
      bool(case and case["exit_reason"] == "AMS_3_STRIKES" and case["triage_verdict"]["esi_level"] <= 2)
      and plays(ev, "ams_alert") == 1, f"exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "เจ็บมากค่ะ"], telemetry={"dominant_current_gesture": "CHEST_PAIN"},
                          final_reply=final)
check("S5", "Structured: clutching chest now + 'เจ็บมาก' -> emergency, alert once",
      bool(case and case["exit_reason"] == "EMERGENCY_INTERRUPT") and plays(ev, "emergency_alert") == 1,
      f"exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET_EN + ["Yes I agree", "I can't stop coughing since yesterday", "4", ""],
                          final_reply=final)
check("S6", "Structured: 'I can't stop coughing since yesterday' -> continues, onset taken from it",
      bool(case and case["exit_reason"] == "NORMAL_COMPLETE" and (case.get("intake") or {}).get("onset")),
      f"exit={case and case['exit_reason']}, intake={case and case.get('intake')}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ยกเลิกค่ะ"], final_reply=final)
check("S7", "Structured: 'ยกเลิกค่ะ' cancels, acknowledgement spoken, no verdict computed",
      bool(case and case["exit_reason"] == "USER_CANCELLED")
      and says(ev, SystemConfig.SPOKEN_PROMPTS["th"]["cancel_ack"]) == 1
      and not any(e == ("LLM_CALL", "final") for e in ev), f"exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET_EN + ["Yes I agree", "can you speak Thai please", "ปวดท้องค่ะ", "5",
                                      "เมื่อวาน", ""], final_reply=final)
check("S8", "Structured: patient asks for Thai mid-interview -> question repeated in Thai, case in Thai",
      bool(case and case["language"] == "th") and says(ev, QUESTIONS["th"]["complaint"]) == 1
      and (case.get("intake") or {}).get("pain_score_0_10") == 5, f"lang={case and case['language']}, "
      f"intake={case and case.get('intake')}")


MORE = SystemConfig.SPOKEN_PROMPTS["th"]["anything_else_more"]
case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัวค่ะ", "6", "เมื่อเช้า", "เจ็บมาตั้งแต่ 6 โมงเช้าค่ะ",
                                   "มีไข้ด้วยค่ะ", "โอเค ไม่มีอะไรแล้วครับ"], final_reply=final)
pt = [t["text"] for t in (case or {}).get("conversation", []) if t["role"] == "patient"]
check("S9", "Structured: 'anything else?' repeats after each extra statement until 'ไม่มีอะไรแล้ว'",
      bool(case and case["exit_reason"] == "NORMAL_COMPLETE") and says(ev, MORE) == 2
      and "มีไข้ด้วยค่ะ" in pt and not unheard(case, ev), f"'more' asked {says(ev, MORE)}x, patient turns={pt}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัวค่ะ", "6", "เมื่อเช้า", "", ""], final_reply=final)
check("S10", "Structured: silence after 'anything else?' -> asked once more, then the case closes",
      bool(case and case["exit_reason"] == "NORMAL_COMPLETE") and says(ev, MORE) == 1,
      f"'more' asked {says(ev, MORE)}x, exit={case and case['exit_reason']}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดหัวค่ะ", "6", "เมื่อเช้า"] + ["อีกอย่างค่ะ"] * 8, final_reply=final)
check("S11", "Structured: endless extra statements are capped (no infinite loop)",
      bool(case and case["exit_reason"] == "NORMAL_COMPLETE")
      and says(ev, MORE) == SystemConfig.ANYTHING_ELSE_MAX_ROUNDS - 1, f"'more' asked {says(ev, MORE)}x")


case, hand, ev, err = run(["สวัสดีครับ", "English please", "Yes I agree", "headache", "5", "since this morning",
                           "that's all"], final_reply=final)
pdpa = [e for e in ev if e == ("PLAY", "pdpa_notice")]
check("G1", "Greets in Thai, patient answers 'English please' -> consent + interview in English",
      bool(case and case["language"] == "en" and case["pdpa_consented"]) and len(pdpa) == 1
      and says(ev, QUESTIONS["en"]["severity"]) == 1, f"lang={case and case['language']}, notices={len(pdpa)}")

case, hand, ev, err = run(["สวัสดีครับ", "ครับ", "Sorry, can you speak English?", "Yes, I agree", "headache", "5",
                           "this morning", "that's all"], final_reply=final)
pdpa = [e for e in ev if e == ("PLAY", "pdpa_notice")]
check("G2", "Language can still change at the consent question: notice repeated in English, then consent",
      bool(case and case["language"] == "en" and case["pdpa_consented"]) and len(pdpa) == 2,
      f"lang={case and case['language']}, notices={len(pdpa)}, handoffs={hand}")

# ---------------------------------------------------------------- hardware fault
case, hand, ev, err = run(GREET, mic_error=True, max_seconds=2.0)
check("H1", "Microphone unplugged -> staff told once (anonymous 'kiosk_fault' event), no silent loop",
      [h["reason"] for h in hand] == ["kiosk_fault"] and set(hand[0]) <= {"event_type", "reason", "event_id"},
      f"handoffs={hand}, FSM exceptions={len(err)}")

case, hand, ev, err = run(GREET + ["ยินยอมค่ะ", "ปวดท้องค่ะ"], fail_after=4, turn_reply=turn, final_reply=final)
check("H2", "Mic fails mid-interview -> the consented case is still sent (SYSTEM_FAULT) + staff alerted",
      bool(case and case["exit_reason"] == "SYSTEM_FAULT" and case["triage_verdict"])
      and any(h["reason"] == "kiosk_fault" for h in hand),
      f"exit={case and case['exit_reason']}, ESI={case and (case['triage_verdict'] or {}).get('esi_level')}, "
      f"handoffs={hand}")

# ---------------------------------------------------------------- report
w = max(len(r[2]) for r in RESULTS)
for sid, status, title, detail in RESULTS:
    print(f"{sid:3} {status}  {title.ljust(w)}\n          -> {detail}")
print(f"\n{sum(r[1] == 'PASS' for r in RESULTS)}/{len(RESULTS)} scenarios behaved safely.")
sys.exit(0 if all(r[1] == "PASS" for r in RESULTS) else 1)
