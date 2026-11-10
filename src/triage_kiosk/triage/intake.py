"""Structured symptom intake: ask each question once, in a fixed order, and never loop.

Order (a short OPQRST-style script, like a triage nurse at the door):
  1. complaint - "What is bothering you today?"           (any answer is accepted)
  2. severity  - "From 0 to 10, how bad is it right now?" (number or mild/moderate/severe words)
  3. onset     - "When did it start?"                      (time words, "2 hours ago", "เมื่อวาน", ...)
  then the kiosk asks "anything else?" and closes the case.

If the patient already said the score or the start time (e.g. "ปวดหัว 8 เต็ม 10 ตั้งแต่เมื่อวาน"),
that question is skipped. A question that gets no usable answer is asked again ONCE with a
clearer wording, then the kiosk moves on ("not stated") - so it can never loop.

The rules are deterministic on purpose: in the first Mac test the free-form LLM questions
repeated themselves (it did not track what had already been answered). The LLM is still
used for the final ESI suggestion and SOAP note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

SLOTS = ("complaint", "severity", "onset")

QUESTIONS = {
    "th": {
        "complaint": "วันนี้มีอาการไม่สบายอะไรมาคะ เล่าให้ดิฉันฟังได้เลยค่ะ",
        "severity": "ถ้าให้คะแนนความเจ็บหรือความไม่สบาย ตั้งแต่ศูนย์ถึงสิบ ตอนนี้ให้กี่คะแนนคะ",
        "onset": "อาการนี้เริ่มเป็นตั้งแต่เมื่อไหร่คะ เช่น กี่ชั่วโมง หรือกี่วันมาแล้วคะ",
    },
    "en": {
        "complaint": "What is bothering you today? Please tell me your main symptom.",
        "severity": "On a scale from zero to ten, how bad is it right now?",
        "onset": "When did it start? For example, how many hours or days ago?",
    },
}
REPROMPTS = {
    "th": {
        "complaint": "ขออภัยค่ะ ช่วยบอกอาการหลักอีกครั้งได้ไหมคะ เช่น ปวดหัว ปวดท้อง หรือเจ็บหน้าอกคะ",
        "severity": "ขอเป็นตัวเลขศูนย์ถึงสิบค่ะ ศูนย์คือไม่เจ็บเลย สิบคือเจ็บมากที่สุด ตอนนี้ให้กี่คะแนนคะ",
        "onset": "ประมาณกี่ชั่วโมง หรือกี่วันแล้วคะ ถ้าไม่แน่ใจ ตอบว่าไม่แน่ใจได้ค่ะ",
    },
    "en": {
        "complaint": "Sorry, could you tell me your main symptom again? For example headache, stomach pain or chest pain.",
        "severity": "Please give a number from zero to ten. Zero is no pain and ten is the worst pain. What number is it now?",
        "onset": "Roughly how many hours or days ago did it start? You can say not sure.",
    },
}


def all_fixed_texts() -> set[str]:
    """Every sentence this module can make the kiosk say (fixed text, no patient data)."""
    out = set()
    for table in (QUESTIONS, REPROMPTS):
        for lang in table.values():
            out.update(lang.values())
    return out


# ---------------------------------------------------------------- number words
_TH_DIGITS = {"ศูนย์": 0, "หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7,
              "แปด": 8, "เก้า": 9, "สิบ": 10}
_EN_DIGITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10}
_NUM_WORD = "|".join(sorted(list(_TH_DIGITS) + list(_EN_DIGITS), key=len, reverse=True))
_NUM = rf"(\d{{1,2}}(?:\.\d)?|{_NUM_WORD})"


def _to_num(tok: str) -> Optional[float]:
    tok = tok.strip().lower()
    if tok in _TH_DIGITS:
        return float(_TH_DIGITS[tok])
    if tok in _EN_DIGITS:
        return float(_EN_DIGITS[tok])
    try:
        return float(tok)
    except ValueError:
        return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


# ---------------------------------------------------------------- severity
_SEV_EXPLICIT = [
    rf"{_NUM}\s*(?:/|out of|เต็ม|จาก|ใน)\s*(?:10|สิบ|ten)",
    rf"{_NUM}\s*(?:คะแนน|points?|score)",
    rf"(?:คะแนน|score|scale|ระดับ)\s*(?:is|of|=|คือ|ประมาณ|ได้)?\s*{_NUM}",
]
_SEV_WORDS = [  # (pattern, label) - only when the severity question was asked; order matters
    (r"ไม่เจ็บ|ไม่ปวด|no pain|doesn'?t hurt|does not hurt|not painful", "none"),
    (r"ไม่ค่อย|ไม่มาก|นิดหน่อย|นิดเดียว|เล็กน้อย|น้อย|mild|a little|slight|not much|not (?:that |too )?bad", "mild"),
    (r"ปานกลาง|พอทน|พอสมควร|moderate|medium|so-so", "moderate"),
    (r"ทนไม่ไหว|มากที่สุด|มาก|รุนแรง|worst|unbearable|severe|terrible|really bad|very bad|a lot", "severe"),
]


def extract_severity(text: str, asked: bool = False) -> tuple[Optional[int], Optional[str]]:
    """Return (score 0-10 or None, label or None).

    Outside the severity question only explicit forms count ("7/10", "7 เต็ม 10", "7 คะแนน"),
    so "2 hours" or "2 days" is never read as a pain score.
    """
    t = _norm(text)
    if not t:
        return None, None
    for pat in _SEV_EXPLICIT:
        m = re.search(pat, t)
        if m:
            n = _to_num(m.group(1))
            if n is not None and 0 <= n <= 10:
                return int(round(n)), _label(n)
    if asked:
        # a bare number, but not one followed by a time unit ("3 วัน", "2 hours")
        nums = [m for m in re.finditer(_NUM, t)
                if not re.match(r"\s*(?:นาที|ชั่วโมง|ชม|วัน|สัปดาห์|อาทิตย์|เดือน|ปี|โมง|min|hour|hr|day|week|month|year|o'?clock|am|pm)",
                                t[m.end():])]
        values = [v for v in (_to_num(m.group(1)) for m in nums) if v is not None and 0 <= v <= 10]
        if len(values) >= 1:
            return int(round(values[0])), _label(values[0])
        for pat, label in _SEV_WORDS:
            if re.search(pat, t):
                return None, label
    return None, None


def _label(n: float) -> str:
    return "none" if n == 0 else "mild" if n <= 3 else "moderate" if n <= 6 else "severe"


# ---------------------------------------------------------------- onset
_TH_UNITS = r"(?:นาที|ชั่วโมง|ชม\.?|วัน|สัปดาห์|อาทิตย์|เดือน|ปี)"
_EN_UNITS = r"(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)"
_ONSET_PATTERNS = [
    rf"{_NUM}\s*{_TH_UNITS}",
    rf"(?:{_NUM}|a|an|few|couple of|several)\s*{_EN_UNITS}",
    r"(?:ครึ่ง|หลาย|ไม่กี่)\s*" + _TH_UNITS,
    r"half an? hour",
    r"เมื่อวานซืน|เมื่อวาน|เมื่อเช้า|เช้านี้|เมื่อคืน|คืนนี้|เมื่อกี้|เมื่อตะกี้|เมื่อบ่าย|บ่ายนี้|เมื่อเย็น|วันนี้|ตอนเช้า|ตอนบ่าย|ตอนเย็น|ตอนกลางคืน|ตอนตื่น",
    rf"{_NUM}\s*โมง|ตี\s*{_NUM}|เที่ยง",
    rf"(?:\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm|o'?clock))",
    r"yesterday|this morning|last night|tonight|today|this afternoon|this evening|just now|an hour ago|since (?:\w+)",
    r"นานแล้ว|ไม่นาน|พักหนึ่ง|เรื้อรัง|a while|long time|recently|chronic|on and off|เป็นๆหายๆ",
]
_ONSET_UNKNOWN = r"ไม่แน่ใจ|ไม่รู้|จำไม่ได้|ไม่ทราบ|not sure|don'?t know|do not know|can'?t remember|no idea"


def extract_onset(text: str, asked: bool = False) -> Optional[str]:
    """Return the words that describe when it started, "unknown" if the patient is not sure, or None."""
    t = _norm(text)
    if not t:
        return None
    for pat in _ONSET_PATTERNS:
        m = re.search(pat, t)
        if m:
            return m.group(0).strip()
    if asked and re.search(_ONSET_UNKNOWN, t):
        return "unknown"
    return None


# ---------------------------------------------------------------- intake state
@dataclass
class StructuredIntake:
    max_asks: int = 2
    complaint: Optional[str] = None
    severity_score: Optional[int] = None
    severity_label: Optional[str] = None
    onset: Optional[str] = None
    asks: dict = field(default_factory=lambda: {s: 0 for s in SLOTS})
    skipped: list = field(default_factory=list)
    unusable_in_a_row: int = 0
    last_asked: Optional[str] = None

    def filled(self, slot: str) -> bool:
        if slot == "complaint":
            return bool(self.complaint)
        if slot == "severity":
            return self.severity_score is not None or self.severity_label is not None
        return self.onset is not None

    def next_question(self, lang: str) -> Optional[tuple[str, str]]:
        """(slot, sentence) for the next thing to ask, or None when the intake is complete."""
        for slot in SLOTS:
            if self.filled(slot) or slot in self.skipped:
                continue
            if self.asks[slot] >= self.max_asks:
                self.skipped.append(slot)          # give up on this one: never loop
                continue
            text = (QUESTIONS if self.asks[slot] == 0 else REPROMPTS)[lang][slot]
            self.asks[slot] += 1
            self.last_asked = slot
            return slot, text
        self.last_asked = None
        return None

    def mark_asked(self, slot: str) -> None:
        """The opening question (pre-recorded or posture prompt) already asked for the complaint."""
        self.asks[slot] += 1
        self.last_asked = slot

    def update(self, utterance: str) -> bool:
        """Read one patient answer. Returns True if it answered the question that was asked."""
        asked = self.last_asked
        text = (utterance or "").strip()
        if not text:
            return False
        before = {s: self.filled(s) for s in SLOTS}

        score, label = extract_severity(text, asked=(asked == "severity"))
        if score is not None and self.severity_score is None:
            self.severity_score, self.severity_label = score, label
        elif label and not self.filled("severity"):
            self.severity_label = label
        onset = extract_onset(text, asked=(asked == "onset"))
        if onset and self.onset is None:
            self.onset = onset
        if asked == "complaint" and not self.complaint:
            self.complaint = text
        elif not self.complaint and asked is None:
            self.complaint = text

        answered = asked is None or (self.filled(asked) and not before.get(asked, False))
        self.unusable_in_a_row = 0 if answered else self.unusable_in_a_row + 1
        return answered

    def as_dict(self) -> dict:
        return {
            "complaint": self.complaint,
            "pain_score_0_10": self.severity_score,
            "severity": self.severity_label,
            "onset": self.onset,
            "not_answered": list(self.skipped),
        }
