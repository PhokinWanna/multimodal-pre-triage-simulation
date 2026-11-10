# Real-world test and demo on a Mac (camera + microphone)

This guide tests the whole product end to end on one Mac: kiosk (camera, mic, speaker,
Whisper, Llama 3.2) → outbox → nurse dashboard. Every test lists what to do, what you should
hear/see, and how to judge PASS/FAIL. Section 5 is a short demo flow for showing the work.

> This is a research prototype, not a medical device. Say so when you demo it.

---

## 1. One-time setup (≈ 30 min, needs internet once)

```bash
cd ~/Desktop/All_Project/multimodal-pre-triage-simulation

# Kiosk environment (Python 3.11)
brew install portaudio                 # needed by PyAudio (microphone)
python3.11 -m venv .venv-kiosk
source .venv-kiosk/bin/activate
python -m pip install -r requirements-kiosk.txt
deactivate

# Dashboard environment
python3.11 -m venv .venv-dashboard
source .venv-dashboard/bin/activate
python -m pip install -r requirements-dashboard.txt
PYTHONPATH=src python -m nurse_dashboard create-admin      # your admin login
PYTHONPATH=src python -m nurse_dashboard create-user       # e.g. "nurse1" for the demo
deactivate

# LLM
#   install the Ollama app from https://ollama.com, open it once, then:
ollama pull llama3.2
```

**Thai voice for the follow-up questions (offline, nothing leaves the Mac):**
System Settings → Accessibility → Spoken Content → System voice → Manage Voices… → Thai →
download **Kanya**. Test it: `say -v Kanya "สวัสดีค่ะ"`.
Without it, the kiosk still works but plays the generic question recording instead of the
LLM's own question (and the transcript shows the generic question, i.e. what was really said).

**Permissions:** the first run asks for Camera and Microphone access for Terminal (or
VS Code/iTerm). Allow both. If you clicked "Don't allow": System Settings → Privacy &
Security → Camera / Microphone → turn on your terminal app, then restart the terminal.

**Whisper model:** the first kiosk start downloads Faster-Whisper "small" (~500 MB).

---

## 2. Start everything (each time, 3 terminal windows)

Pick a kiosk key once (any random text), e.g. `demo-kiosk-key-2026`.

**Terminal 1 – dashboard**

```bash
cd ~/Desktop/All_Project/multimodal-pre-triage-simulation
source .venv-dashboard/bin/activate
export DASHBOARD_KIOSK_KEY=demo-kiosk-key-2026
PYTHONPATH=src python -m uvicorn nurse_dashboard.app:app --port 8000
```

Open http://127.0.0.1:8000 in the browser, log in as `nurse1`, and click **Turn on alert**
(browsers only play sound after a click).

**Terminal 2 – pre-flight check** (must end with `0 FAIL`)

```bash
cd ~/Desktop/All_Project/multimodal-pre-triage-simulation
source .venv-kiosk/bin/activate
export NURSE_DASHBOARD_KEY=demo-kiosk-key-2026
PYTHONPATH=src python scripts/checks/kiosk_preflight.py --full
```

`--full` records 4 s from the mic (speak when asked), plays the greeting, speaks one Thai and
one English test question, loads Whisper (shows what it heard) and times the LLM.

| Line                            | What FAIL means / fix                                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| import pyaudio                  | `brew install portaudio`, then `pip install PyAudio` again                                                              |
| camera index 0                  | camera permission, or another app (Zoom/FaceTime) is using it                                                               |
| microphone recording level WARN | mic muted / no permission / wrong input in System Settings → Sound → Input                                                |
| Ollama model                    | open the Ollama app; `ollama pull llama3.2`                                                                               |
| LLM latency FAIL                | the Mac is slower than the timeout; set `export OLLAMA_TURN_TIMEOUT=15 OLLAMA_FINAL_TIMEOUT=30` before starting the kiosk |
| nurse dashboard reachable       | Terminal 1 is not running                                                                                                   |
| offline th voice WARN           | download Kanya (section 1)                                                                                                  |

**Terminal 3 – kiosk**

```bash
cd ~/Desktop/All_Project/multimodal-pre-triage-simulation
source .venv-kiosk/bin/activate
export NURSE_DASHBOARD_KEY=demo-kiosk-key-2026
PYTHONPATH=src python -m triage_kiosk.main
```

Two windows open (patient view + skeleton view). Press **q** in a window to stop.
Keep Terminal 3 visible: it prints every state, what Whisper heard, the LLM decision and the
timing lines `[STT] …` and `[TELEMETRY]: Listen+STT … | LLM … | TTS …` (section 6).

**Set-up for good results:** sit 0.5–1 m from the camera with your head and both hands
visible, even front light (no window behind you), quiet room, speak clearly after the kiosk
finishes talking (it does not listen while speaking).

**Between tests:** after a case finishes, the kiosk waits 15 s (or until you step out of
the camera view) before it starts a new encounter, so a "thank you" does not start a new case.

---

## 3. Test cases

Mark each one PASS/FAIL. "Dashboard" means the queue page in the browser.

### A. Normal flow

The interview is a fixed, step-by-step script: **main complaint → pain score 0–10 → when it
started → "anything else?"**. Each question is asked once; if the answer is unclear it is asked
once more in clearer words, then the kiosk moves on (it never loops). A question you already
answered is skipped.

**T1 – Thai normal case**
1. Sit in front of the camera. Say **"สวัสดีค่ะ"**.
2. Greeting plays → say **"ภาษาไทยค่ะ"**.
3. PDPA notice plays → say **"ยินยอมค่ะ"**.
4. Kiosk asks what is wrong → **"ปวดท้องค่ะ"**.
5. Kanya voice: "…ตั้งแต่ศูนย์ถึงสิบ ตอนนี้ให้กี่คะแนนคะ" → **"หกค่ะ"** (or "6").
6. Kanya voice: "อาการนี้เริ่มเป็นตั้งแต่เมื่อไหร่คะ…" → **"ตั้งแต่เมื่อวานค่ะ"**.
7. "มีอะไรเพิ่มเติมอีกไหมคะ" → **"ไม่มีแล้วค่ะ"** → the wrap-up message plays while the
   LLM writes the note in the background.

PASS if: no question is repeated; the case reaches the dashboard with the transcript,
**Pain score 6/10 and Onset** under "Situation", a suggested ESI level, the SOAP note (in
English) and the photo; every kiosk line in the transcript is something you really heard.

**T2 – English normal case**: "Hello" → "English please" → "Yes, I agree" →
"I have a headache" → "five" → "since this morning" → "That's all". PASS like T1.

**T2b – All in one answer**: after consent say **"ปวดหัว 8 เต็ม 10 ตั้งแต่เมื่อวานค่ะ"**. PASS if the
kiosk does **not** ask for the score or the start time and goes straight to "anything else".
Because the patient rated the pain 8/10, the suggested level is ESI 3 or more urgent.

**T2c – Unclear score**: answer the score question with **"อืม"**. PASS if it asks once more
("ขอเป็นตัวเลขศูนย์ถึงสิบค่ะ…"); answer "ไม่รู้" again and it moves on to the start-time question
(no loop). The case shows "not stated" for the score.

**T2d – Change language**: start in English, after consent say **"Can you speak Thai please"**.
PASS if the question is repeated in Thai and the rest continues in Thai.

**T3 – Silence closes the encounter**: consent, then say nothing. Kiosk says "still
listening", then "anything else", then closes. PASS if the case reaches the dashboard and the
kiosk goes back to standby.

### B. Consent (PDPA)

**T4 – Refusal**: at the PDPA question say **"ไม่ยินยอมค่ะ"**. PASS if the kiosk tells you to
go to the nurse desk, the dashboard shows the banner "A patient at the kiosk needs a nurse ·
Declined consent", and **no case/photo/transcript** is stored.

**T5 – Unclear / silent**: at the PDPA question stay silent. The kiosk re-asks ("กรุณาตอบว่า
ยินยอม หรือ ไม่ยินยอม"). Stay silent again. PASS if it routes you to the nurse (banner
"Consent answer unclear") and stores nothing. *Silence is never treated as consent.*

**T6 – Polite yes with "ไม่"**: answer **"ไม่มีปัญหาค่ะ"**. PASS if it counts as consent.

### C. Emergency and posture

**T7 – Emergency words**: after consent say **"แน่นหน้าอกมาก หายใจไม่ค่อยออก"**. PASS if the
emergency alert plays **immediately**, it plays **once**, and the dashboard shows an urgent case
(ESI 1 or 2, alert sound) within about a second.

**T8 – Chest clutching (camera)**: after consent, put your hand flat on the middle of your
chest and keep it there for at least 3 s while saying **"เจ็บหน้าอกค่ะ เป็นมา 2 ชั่วโมง"**. The
skeleton window should show the chest gesture. PASS if the case is ESI 1–2 and the objective
notes show chest clutching. (If you hold your chest *before* the interview starts, the kiosk
asks about chest pain first.)

**T9 – Abdomen / head posture**: same as T8 holding your stomach ("ปวดท้อง") or your head
("ปวดหัว"). PASS if the posture is listed on the case.

### D. Words that used to cause false triggers

**T10** – "I can't stop coughing since yesterday" → the interview continues (not cancelled).
**T11** – "ยาความดันหมดแล้วค่ะ เวียนหัว" → the interview continues (not ended).
**T12** – "ยกเลิกค่ะ" → the kiosk stops and resets (this one *should* cancel).

### E. Failure handling (shows the system is safe when parts break)

**T13 – LLM down**: quit the Ollama app, run T1 again. PASS if the kiosk still finishes, the
case arrives on the dashboard and is marked as a rule-based (fallback) suggestion. Re-open
Ollama afterwards.

**T14 – Dashboard down**: stop Terminal 1 (Ctrl+C), run T1. The kiosk finishes normally and
Terminal 3 says the dashboard is unreachable. Start Terminal 1 again. PASS if the case appears
within ~30 s (it waited safely in the outbox on disk).

**T15 – Microphone fault** (only with an external/USB mic): unplug it while the kiosk is in
standby with you in front of it. PASS if within ~15 s the alert plays (if the speaker works)
and the dashboard shows "Kiosk fault (microphone / speech / software)". Plug it back in and
restart the kiosk.

---

## 4. Record the results

Write down for each test: PASS/FAIL, the `[TELEMETRY]` turnaround times from Terminal 3, and
anything Whisper misheard. Screenshots of the dashboard (queue + one case page) and a short
screen recording of T1 and T7 make good evidence.

---

## 5. Suggested 5-minute demo (for HR / interview)

1. **Architecture (30 s)** – one sentence each: camera → MediaPipe posture + skin CNN;
   mic → Faster-Whisper; local Llama 3.2 asks follow-up questions and suggests an ESI level;
   everything runs **on the Mac** (privacy / PDPA); a nurse dashboard receives the case and the
   nurse makes the final decision (human-in-the-loop).
2. **T1 live (2 min)** – a normal Thai patient; switch to the dashboard and open the case.
3. **T7 or T8 live (1 min)** – emergency: alert plays immediately, urgent case with sound.
4. **T4 (30 s)** – consent refused: nothing stored, nurse is called.
5. **Reliability (1 min)** – mention T13/T14 (works when the LLM or dashboard is down), the
   165 automated tests and the 29-scenario state-machine simulation
   (`PYTHONPATH=src python scripts/checks/kiosk_fsm_sim.py`).

Good things to say honestly: it is a prototype; ESI suggestions are checked by a safety floor
(hard signals such as chest clutching or emergency words can never be down-graded by the
LLM); the nurse always makes the final call; accuracy is still being evaluated (M3 experiment).

---

## 6. Reading the log lines

```
[STT] waited+recorded 6.1 s (4.3 s audio), Whisper decode 1.2 s, 54 chars
[Intake]: answered=True {'complaint': 'ปวดท้องค่ะ', 'pain_score_0_10': 6, ...}
[TELEMETRY]: Listen+STT: 7.40s | LLM: 0.00s (structured) | TTS: 3.90s | Turnaround: 11.4s
```

* **waited+recorded** is mostly *you talking* plus the 1.2 s of silence that ends a reply.
  If you see "Reply hit the 12 s limit", room noise keeps the microphone "on": restart the
  kiosk with `export MIC_ENERGY_THRESHOLD=600` (try 400–1000).
* **Whisper decode** should be 1–4 s. **LLM** is 0 during the interview (fixed questions);
  the LLM runs once at the end for the ESI + note, while the wrap-up message is playing.
  **TTS** is the time the voice spends speaking.
* `export KIOSK_LOG_TRANSCRIPTS=1` prints what Whisper heard (`[Patient]: "…"`). For testing
  only: it shows patient words in the terminal.
* `export INTERVIEW_MODE=llm` switches back to the original free-form LLM questions (for comparison).
* If one turn takes over 60 s, the kiosk prints a stack trace of every thread
  ("Thread 0x… most recent call first"). Copy that block: it shows exactly where it froze.

## 7. Troubleshooting

| Problem                                                | Fix                                                                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Kiosk never starts talking                             | It needs to see a person (camera window) **and** hear speech; sit closer, say "สวัสดีค่ะ" |
| Whisper mishears Thai                                  | Speak slower and closer to the mic; reduce background noise                                              |
| Long pause after you answer                            | Check `Whisper decode` in the `[STT]` line; close other heavy apps                                    |
| Follow-up question is always the same generic sentence | Thai voice not installed → section 1                                                                    |
| Dashboard 401 in Terminal 3                            | `NURSE_DASHBOARD_KEY` (kiosk) ≠ `DASHBOARD_KIOSK_KEY` (dashboard)                                   |
| No sound from the dashboard                            | Click "Turn on alert" in the browser                                                                     |
| `Could not open camera device 0`                     | Close Zoom/FaceTime/Photo Booth; check camera permission                                                 |
| Old cases appear after restart                         | Normal: the outbox re-sends anything not yet delivered                                                   |
