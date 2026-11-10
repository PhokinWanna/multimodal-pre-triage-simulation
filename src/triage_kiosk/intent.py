"""Rule-based intent matching for short patient replies (consent, cancel, stop, wake word).

Plain substring matching caused false triggers in the kiosk check (``kiosk_fsm_sim.py``):
"no" inside "know", "ไม่" inside "ไม่มีปัญหา", "stop" inside "can't stop coughing" and
"หมดแล้ว" inside "ยาหมดแล้ว". The rules here are:

* English is matched on whole words / phrases (regex word boundaries), with negations checked.
* Thai has no spaces, so Thai is matched on phrases; negative phrases are checked before
  positive ones, and known positive phrases that contain "ไม่" (ไม่มีปัญหา, ไม่เป็นไร) are
  removed first.
* "End of interview" phrases count only when they are (almost) the whole utterance.
"""

from __future__ import annotations

import re

YES, NO, UNCLEAR = "YES", "NO", "UNCLEAR"

# Polite particles / fillers removed before matching short Thai replies.
_TH_PARTICLES = ["ค่ะ", "คะ", "ครับ", "คับ", "ค่า", "จ้ะ", "จ้า", "นะ", "จ๊ะ", "ฮะ", "เลย"]

# ---------------------------------------------------------------- consent
_TH_CONSENT_NO = ["ไม่ยินยอม", "ไม่ยอม", "ไม่ตกลง", "ไม่อนุญาต", "ไม่ให้", "ไม่เอา", "ไม่สะดวก",
                  "ไม่ต้องการ", "ไม่โอเค", "ไม่ได้", "ปฏิเสธ"]
_TH_CONSENT_YES = ["ยินยอม", "ยินดี", "ตกลง", "อนุญาต", "ได้", "โอเค", "ใช่", "ยอมรับ", "ไม่มีปัญหา",
                   "ไม่ขัดข้อง", "เห็นด้วย", "เอา", "รับทราบ", "เชิญ"]
_TH_PARTICLE_ONLY_YES = {"ค่ะ", "คะ", "ครับ", "คับ", "ค่า", "จ้ะ", "จ้า", "ฮะ", "อือ", "อืม", "เออ"}
_TH_POSITIVE_WITH_MAI = ["ไม่มีปัญหา", "ไม่ขัดข้อง", "ไม่เป็นไร"]

_EN_CONSENT_NO = [r"\bno\b(?!\s+problem)", r"\bnope\b", r"\bdisagree\b", r"\brefuse\b", r"\bdecline\b",
                  r"\bdon'?t (agree|consent|accept|want)\b", r"\bdo not (agree|consent|accept|want)\b",
                  r"\bnot (agree|consent|okay|ok)\b", r"\bi object\b"]
_EN_CONSENT_YES = [r"\byes\b", r"\byeah\b", r"\byep\b", r"\bagree\b", r"\bi consent\b", r"\bconsent\b",
                   r"\baccept\b", r"\bokay\b", r"\bok\b", r"\bsure\b", r"\bfine\b", r"\bno problem\b",
                   r"\bof course\b", r"\bgo ahead\b", r"\balright\b", r"\buh[- ]?huh\b", r"\byup\b",
                   r"\bplease do\b", r"\bthat'?s fine\b"]


def _has_thai(text: str) -> bool:
    return any("฀" <= ch <= "๿" for ch in text)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _strip_th(text: str) -> str:
    t = text.replace(" ", "")
    for p in _TH_PARTICLES:
        t = t.replace(p, "")
    return t


def _any_re(patterns, text) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify_consent(text: str) -> str:
    """Return YES, NO or UNCLEAR. Silence or anything unclear is UNCLEAR (never consent)."""
    t = _norm(text)
    if not t:
        return UNCLEAR
    th = t.replace(" ", "")
    for p in _TH_POSITIVE_WITH_MAI:
        th = th.replace(p, " OKPHRASE ")
    th_no = (any(p in th for p in _TH_CONSENT_NO) or _strip_th(th) == "ไม่"
             or any("ไม่" + w in th for w in _TH_CONSENT_YES))
    th_yes = ("OKPHRASE" in th or any(p in th for p in _TH_CONSENT_YES)
              or th.strip(" .!") in _TH_PARTICLE_ONLY_YES)
    en_no = _any_re(_EN_CONSENT_NO, t)
    en_yes = _any_re(_EN_CONSENT_YES, t)
    no, yes = th_no or en_no, th_yes or en_yes
    if no and not yes:
        return NO
    if yes and not no:
        return YES
    if no and yes:
        # e.g. "ไม่ยินยอม" also contains "ยินยอม": the negative phrase wins.
        return NO
    return UNCLEAR


# ---------------------------------------------------------------- cancel
_TH_CANCEL = ["ยกเลิก", "ไม่เอาแล้ว", "ไม่ทำแล้ว", "ไม่คุยแล้ว", "ขอออก", "ไม่ตรวจแล้ว"]
_EN_CANCEL = [r"\bcancel\b", r"\bi want to (stop|quit|leave)\b", r"\bstop (this|the interview|talking|asking)\b",
              r"\bi'?m leaving\b", r"\bnever ?mind\b", r"\blet me go\b", r"^\s*stop\s*[.!]?\s*$", r"\bquit\b"]
_EN_NEGATED_STOP = r"\b(can'?t|cannot|won'?t|doesn'?t|does not|didn'?t|not|never)\s+(\w+\s+)?stop"


def is_cancel_request(text: str) -> bool:
    t = _norm(text)
    if not t:
        return False
    if any(p in t.replace(" ", "") for p in _TH_CANCEL):
        return True
    if re.search(_EN_NEGATED_STOP, t) and not re.search(r"\bcancel\b", t):
        return False
    return _any_re(_EN_CANCEL, t)


# ---------------------------------------------------------------- end of interview
_TH_STOP = ["หมดแล้ว", "แค่นี้", "ไม่มีแล้ว", "พอแล้ว", "พอแค่นี้", "เสร็จแล้ว", "ไม่มีอะไรเพิ่ม", "มีเท่านี้",
            "ไม่มีอะไรแล้ว", "ไม่มี"]
_EN_STOP = ["that's all", "thats all", "that is all", "nothing else", "no more", "i'm done", "im done",
            "i am done", "that's it", "thats it", "i'm good", "im good", "nothing", "done", "finished", "enough",
            "i'm fine", "im fine", "nope", "no thanks", "no thank you", "not really", "all good"]
_EN_FILLERS = ["thank you", "thanks", "yes", "no", "okay", "ok", "um", "uh", "well", "i think", "so"]
_TH_SLACK = 4      # characters allowed besides the phrase (e.g. "ขอบคุณ" is removed first)


def is_end_of_interview(text: str) -> bool:
    """True only when the reply is essentially just "that's all" / "หมดแล้วค่ะ"."""
    t = _norm(text)
    if not t:
        return False
    if _has_thai(t):
        s = _strip_th(t).replace("ขอบคุณ", "")
        return any(s == p or ((s.startswith(p) or s.endswith(p)) and len(s) - len(p) <= _TH_SLACK)
                   for p in _TH_STOP)
    s = re.sub(r"[^\w' ]", " ", t)
    for f in sorted(_EN_FILLERS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(f)}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s in _EN_STOP


# ---------------------------------------------------------------- wake word
def find_wake_word(text: str, words_th, words_en) -> tuple[bool, str]:
    t = _norm(text)
    if any(w in t for w in words_th):
        return True, "th"
    if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words_en):
        return True, "en"
    return False, "th"


# ---------------------------------------------------------------- language request
def requested_language(text: str) -> str | None:
    """'en' / 'th' when the patient asks to change language, else None."""
    t = _norm(text)
    if re.search(r"อังกฤษ|อิงลิช|อิงลิส|อิ้งลิช|เอ็นลิช|อิงลิซ|\benglish\b|\beng\b", t):
        return "en"
    if re.search(r"ภาษาไทย|พูดไทย|^ไทย|\bthai\b", t):
        return "th"
    return None


def choose_language(reply: str, detected: str | None, confidence: float, current: str,
                    min_confidence: float = 0.6) -> str:
    """Language for the rest of the encounter, from the answer to "Thai or English?".

    An explicit request wins ("English please", "ภาษาไทย", Whisper's Thai spelling "อิงลิช");
    otherwise the language the person actually spoke, if Whisper is reasonably sure.
    """
    asked = requested_language(reply)
    if asked:
        return asked
    if detected in ("th", "en") and confidence >= min_confidence and (reply or "").strip():
        return detected
    return current


# ---------------------------------------------------------------- finished talking ("anything else?")
_NURSE_REQUEST = [r"อยาก(?:เจอ|พบ|คุยกับ)(?:พยาบาล|หมอ|แพทย์)", r"ขอ(?:เจอ|พบ|คุยกับ)(?:พยาบาล|หมอ|แพทย์)",
                  r"\b(?:see|talk to|speak to|call) (?:a |the )?(?:nurse|doctor)\b"]
_BARE_NO = {"no", "nope", "no no", "ไม่", "ไม่มี", "ไม่ค่ะ", "ไม่ครับ"}


def is_ready_to_finish(text: str) -> bool:
    """Answer to "anything else?" that means "no, I'm finished" (or "let me see the nurse")."""
    t = _norm(text)
    if not t:
        return False
    if is_end_of_interview(t):
        return True
    if re.sub(r"[^\w\s]", "", t).strip() in _BARE_NO or _strip_th(t) in {"ไม่", "ไม่มี", "ไม่มีอะไร"}:
        return True
    compact = t.replace(" ", "")
    return any(re.search(p, compact if not p.startswith("\\b") else t) for p in _NURSE_REQUEST)
