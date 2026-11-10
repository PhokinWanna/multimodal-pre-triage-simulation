"""Reasoning engine: LLM-driven symptom interview and ESI scoring.

Talks to a local Llama 3.2 model through Ollama to choose the kiosk's next
action each turn and to propose a final ESI level, with deterministic
rule-based fallbacks when the LLM is unavailable or times out.
"""

import logging
import json
import re
import time
import requests
from ..config import SystemConfig
from ..intent import is_cancel_request, is_end_of_interview

logger = logging.getLogger(__name__)

class ReasoningEngine:
    LLM_TURN_ACTIONS = {"CONTINUE_TRIAGE", "SWITCH_LANGUAGE", "CANCEL_AND_EXIT", "CALL_HUMAN_NURSE"}

    def __init__(self):
        self.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"
        self.warmup()

    def warmup(self):
        """Pre-warms Llama 3.2 in VRAM on boot with a 15-second cold-load runway."""
        logger.info("Pre-warming Llama 3.2 in VRAM (Pre-Flight Test)...")
        for model in dict.fromkeys([SystemConfig.OLLAMA_MODEL, SystemConfig.OLLAMA_TURN_MODEL]):
            payload = {"model": model, "prompt": "ok", "stream": False,
                       "options": {"num_predict": 1}, "keep_alive": -1}
            try:
                # up to 60 s: the first load after a reboot can take ~20-30 s on the M1
                requests.post(self.endpoint, json=payload, timeout=60.0)
                logger.info(f"{model} is hot and pinned in memory.")
            except Exception as e:
                logger.warning(f"Warmup Notice: Ollama not reachable for {model}: {e}")

    # -------------------------------------------------------------------------
    # 1. STOPPING INTENT DETECTOR
    # -------------------------------------------------------------------------
    @staticmethod
    def is_stopping_phrase(utterance: str, lang: str = "th") -> bool:
        """True only when the reply is essentially "that's all" (see ``intent.is_end_of_interview``)."""
        return is_end_of_interview(utterance)

    # -------------------------------------------------------------------------
    # 2. MID-SESSION ACUTE PREEMPTION CHECK (EMERGENCY BRAKE)
    # -------------------------------------------------------------------------
    @staticmethod
    def check_acute_preemption(utterance: str, visual_signs: dict) -> tuple[bool, int]:
        """Scans for sudden life-threatening collapse mid-case."""
        text = utterance.lower()
        chest_v = visual_signs.get("chest_clutching", False)

        critical_words = [
            "จะตาย", "หายใจไม่ออก", "หมดสติ", "วูบ", "แน่นหน้าอกมาก", "ทนไม่ไหว",
            "cannot breathe", "passing out", "heart attack", "unconscious"
        ]
        has_critical_vocal = any(w in text for w in critical_words)

        if "หมดสติ" in text or "unconscious" in text or "cannot breathe" in text:
            return True, 1

        if chest_v or has_critical_vocal:
            return True, 2

        return False, 0

    @staticmethod
    def is_live_chest_emergency(utterance: str, dominant_gesture: str) -> bool:
        """Patient is clutching the chest RIGHT NOW and describes the pain as severe."""
        text = (utterance or "").lower()
        return dominant_gesture == "CHEST_PAIN" and any(
            k in text for k in ["เจ็บมาก", "แน่นมาก", "ปวดมาก", "severe", "crushing", "really bad", "very bad"])

    # -------------------------------------------------------------------------
    # 3. PROACTIVE VISUAL INITIATION PROMPT
    # -------------------------------------------------------------------------
    @staticmethod
    def get_proactive_visual_prompt(dominant_gesture: str, lang: str = "th") -> str:
        """Generates an opening inquiry based on active visual clutching."""
        if dominant_gesture == "HEAD_PAIN":
            return "ดิฉันสังเกตเห็นว่าคุณกำลังจับบริเวณศีรษะ กำลังมีอาการปวดหัวใช่ไหมคะ? เล่าอาการให้ดิฉันฟังได้เลยนะคะ" if lang == "th" else \
                   "I notice you are holding your head. Are you experiencing head pain? Please tell me what you are feeling."
        elif dominant_gesture == "CHEST_PAIN":
            return "ดิฉันสังเกตเห็นคุณกุมหน้าอกอยู่ ตอนนี้รู้สึกแน่นหรือเจ็บหน้าอกมากไหมคะ? บอกดิฉันได้เลยนะคะ" if lang == "th" else \
                   "I see you are clutching your chest. Are you experiencing acute chest pain or tightness? Please let me know."
        elif dominant_gesture == "ABDOMINAL_PAIN":
            return "ดิฉันสังเกตเห็นคุณกุมบริเวณท้องอยู่ ตอนนี้มีอาการปวดท้องรุนแรงไหมคะ?" if lang == "th" else \
                   "I notice you are holding your abdomen. Are you having severe stomach pain?"
        return ""

    # -------------------------------------------------------------------------
    # 4. AGENTIC ACTION DISPATCHER (Llama 3.2 In Command)
    # -------------------------------------------------------------------------
    def dispatch_agent_turn(self, utterance: str, history: list, visual_telemetry: dict, current_strikes: int, active_lang: str) -> dict:
        """
        Parses patient statement with repeat penalty and multi-turn completion gating.
        """
        text = utterance.strip().lower()
        dom_gesture = visual_telemetry.get("dominant_current_gesture", "NONE")

        # 1. Fast Rule Checks: Cancellation or Life Threat
        if is_cancel_request(text):
            cancel_resp = "รับทราบค่ะ ขออภัยที่ทำให้เสียเวลา ดิฉันจะส่งเรื่องให้คุณพบแพทย์ทันทีนะคะ" if active_lang == "th" else \
                          "Understood. I apologize for the delay. Please proceed directly to see the doctor."
            return {
                "action": "CANCEL_AND_EXIT",
                "target_language": active_lang,
                "spoken_response": cancel_resp,
                "strikes": current_strikes,
                "llm_latency_sec": 0.0
            }

        critical_words = ["จะตาย", "หายใจไม่ออก", "หมดสติ", "วูบ", "แน่นหน้าอกมาก", "ทนไม่ไหว", "cannot breathe", "passing out", "heart attack", "unconscious"]
        if any(w in text for w in critical_words) or (dom_gesture == "CHEST_PAIN" and any(k in text for k in ["เจ็บมาก", "แน่นมาก", "severe", "crushing"])):
            alert_resp = "ระบบตรวจพบอาการฉุกเฉินวิกฤต กรุณานั่งนิ่งๆ สักครู่นะคะ กำลังส่งสัญญาณแจ้งเตือนทีมแพทย์ทันทีค่ะ" if active_lang == "th" else \
                         "Critical emergency detected. Please remain seated while I alert the resuscitation medical team immediately."
            return {
                "action": "EMERGENCY_ABORT",
                "target_language": active_lang,
                "spoken_response": alert_resp,
                "strikes": current_strikes,
                "llm_latency_sec": 0.0
            }

        # 2. Clinical Information Completeness Check
        patient_turns = [turn for turn in history if turn.get("role") == "patient"]
        conversation_context = " ".join([t.get("text", "") for t in patient_turns])
        combined_text = f"{conversation_context} {utterance}".lower()

        has_timing = any(w in combined_text for w in ["เมื่อวาน", "กี่โมง", "โมง", "วัน", "ชั่วโมง", "นาที", "เช้า", "yesterday", "hours", "days", "since", "morning"])
        has_severity = any(w in combined_text for w in ["มาก", "น้อย", "เต็ม 10", "10", "3 เต็ม 10", "พอทน", "severe", "mild", "moderate", "scale", "out of 10"])

        # Stop asking only if Onset + Severity are answered AND at least 2 dialogue rounds occurred
        if has_timing and has_severity and len(patient_turns) >= 2:
            logger.info("[Clinical Check]: Onset and Severity gathered across turns. Triggering TRIAGE_COMPLETE.")
            complete_resp = "รับทราบข้อมูลครบถ้วนค่ะ มีอาการผิดปกติอื่นเพิ่มเติมอีกไหมคะ?" if active_lang == "th" else \
                            "I have noted your symptoms and severity. Is there anything else you would like to add?"
            return {
                "action": "TRIAGE_COMPLETE",
                "target_language": active_lang,
                "is_off_topic": False,
                "spoken_response": complete_resp,
                "strikes": current_strikes,
                "llm_latency_sec": 0.0
            }

        # 3. Strict Language Separation Directives
        if active_lang == "en":
            persona_rules = """LANGUAGE DIRECTIVE:
- You MUST respond 100% in plain ENGLISH.
- DO NOT use any Thai characters or words.
- Tone: Empathetic, polite female triage nurse.
- Max 15 words."""
            schema_spoken = "<One short sentence in English only>"
        else:
            persona_rules = """LANGUAGE DIRECTIVE:
- คุณต้องตอบเป็นภาษาไทย 100% เท่านั้น
- ใช้สรรพนาม 'ดิฉัน' และลงท้ายด้วย 'ค่ะ' หรือ 'นะคะ' เท่านั้น
- ห้ามใช้คำว่า 'ครับ' เด็ดขาด
- ห้ามพูดประโยคซ้ำซาก ห้ามถามซ้ำคำถามเดิม
- ความยาวไม่เกิน 15 คำ"""
            schema_spoken = "<ประโยคภาษาไทยสั้นๆ สุภาพ ไม่ซ้ำซาก ลงท้ายด้วย ค่ะ/นะคะ>"

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an autonomous AI Reception Nurse Agent at an emergency triage kiosk.
{persona_rules}

CLINICAL RULES:
1. If patient requests to change language, set action="SWITCH_LANGUAGE".
2. If patient describes symptoms, ask ONE concise follow-up question (Onset time or Pain scale 1-10).
3. NEVER state or guess disease names (e.g. NEVER ask if they have a heart attack).
4. DO NOT repeat what the patient said multiple times. Ask directly and politely.
5. If patient is off-topic, briefly acknowledge in 4 words and steer back to symptoms.

Output ONLY valid JSON:
{{
  "action": "<CONTINUE_TRIAGE|SWITCH_LANGUAGE|CANCEL_AND_EXIT|CALL_HUMAN_NURSE>",
  "target_language": "<th|en>",
  "is_off_topic": <true|false>,
  "spoken_response": "{schema_spoken}"
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
[ACTIVE LANGUAGE]: {active_lang}
[PATIENT STATEMENT]: "{utterance}"
[DOMINANT OBSERVED GESTURE]: {dom_gesture}
[PRIOR DIALOGUE]: "{conversation_context}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": SystemConfig.OLLAMA_TURN_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 85,      # Runway for JSON + sentence; sub-0.5s execution
                "repeat_penalty": 1.25  # Blocks phrase looping
            },
            "keep_alive": -1
        }

        try:
            t0 = time.time()
            res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TURN_TIMEOUT)
            llm_duration = time.time() - t0

            if res.status_code == 200:
                raw_response = res.json().get("response", "").strip()

                # JSON Auto-Repair Guardrail
                if not raw_response.endswith("}"):
                    if '"' in raw_response and raw_response.count('"') % 2 != 0:
                        raw_response += '"'
                    raw_response += "}"

                frame = json.loads(raw_response)
                if not isinstance(frame, dict):
                    raise ValueError("LLM turn output is not a JSON object")
                frame["llm_latency_sec"] = round(llm_duration, 2)

                # Guardrails: the LLM may only choose these actions, and may not end the
                # case by itself (cancel) unless the patient actually asked to cancel.
                if frame.get("action") not in self.LLM_TURN_ACTIONS:
                    frame["action"] = "CONTINUE_TRIAGE"
                if frame["action"] == "CANCEL_AND_EXIT":
                    frame["action"] = "CONTINUE_TRIAGE"
                if frame.get("target_language") not in ("th", "en"):
                    frame["target_language"] = active_lang
                if frame["action"] == "SWITCH_LANGUAGE" and frame["target_language"] == active_lang:
                    frame["action"] = "CONTINUE_TRIAGE"
                if not str(frame.get("spoken_response") or "").strip():
                    raise ValueError("LLM turn output has no spoken_response")

                if frame.get("target_language") == "th" or active_lang == "th":
                    frame["spoken_response"] = frame.get("spoken_response", "").replace("ครับ", "ค่ะ")

                if frame.get("is_off_topic", False):
                    new_strikes = current_strikes + 1
                    frame["strikes"] = new_strikes
                    if new_strikes >= getattr(SystemConfig, "MAX_OFFTOPIC_STRIKES", 3):
                        frame["action"] = "TRIGGER_AMS"
                        frame["spoken_response"] = SystemConfig.SCRIPT_TEXTS[active_lang]["ams_alert"]
                else:
                    frame["strikes"] = current_strikes

                return frame
        except Exception as e:
            logger.error(f"Agent Dispatch Exception: {e}")

        # Deterministic Fast Fallback
        fallback_msg = "เป็นมานานกี่ชั่วโมงแล้วคะ และเจ็บมากไหม เต็ม 10 ให้เท่าไหร่คะ?" if active_lang == "th" else \
                       "How long have you had this symptom, and how severe is the pain on a scale of 1 to 10?"
        return {
            "action": "CONTINUE_TRIAGE",
            "target_language": active_lang,
            "is_off_topic": False,
            "spoken_response": fallback_msg,
            "strikes": current_strikes,
            "llm_latency_sec": 0.0
        }

    # -------------------------------------------------------------------------
    # 5. SCENARIO A / B / C CROSS-VALIDATION MATRIX
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_scenario(verbal_transcript: str, visual_signs: dict) -> dict:
        """Cross-validates kinetic postures against stated complaints."""
        text = verbal_transcript.lower()
        chest_v = visual_signs.get("chest_clutching", False)
        abdo_v = visual_signs.get("abdominal_clutching", False)
        head_v = visual_signs.get("head_clutching", False)

        has_chest_vocal = any(k in text for k in ["chest", "heart", "เจ็บหน้าอก", "แน่นหน้าอก", "หัวใจ"])
        has_abdo_vocal = any(k in text for k in ["stomach", "abdomen", "belly", "ปวดท้อง", "เสียดท้อง", "จุกแน่น"])
        has_head_vocal = any(k in text for k in ["headache", "dizzy", "stroke", "ปวดหัว", "เวียนหัว", "มึนหัว", "ศีรษะ"])
        has_denial = any(k in text for k in ["fine", "nothing", "okay", "no pain", "ไม่เป็นไร", "สบายดี", "นิดหน่อย"])

        if (chest_v and has_chest_vocal) or (abdo_v and has_abdo_vocal) or (head_v and has_head_vocal):
            return {
                "scenario_type": "SCENARIO_A_CONFIRMED",
                "clinical_flag": "HIGH_CONFIDENCE_ALIGNMENT",
                "notes": "Patient physical pain posturing directly corroborates stated complaint."
            }

        if (chest_v or abdo_v or head_v) and (has_denial or len(text) == 0):
            return {
                "scenario_type": "SCENARIO_C_CLINICAL_CONFLICT",
                "clinical_flag": "SAFETY_OVERRIDE_STOIC_PATIENT",
                "notes": "Patient clutching vital pain zone while verbally minimizing symptoms. Stoic distress suspected."
            }

        return {
            "scenario_type": "SCENARIO_B_SUBJECTIVE_ONLY",
            "clinical_flag": "STANDARD_VERBAL_COMPLAINT",
            "notes": "Symptoms reported verbally without active kinetic clutching."
        }

    # -------------------------------------------------------------------------
    # 6. FINAL CLINICAL SYNTHESIS
    # -------------------------------------------------------------------------
    def evaluate_final_case(self, conversation_history: list, visual_signs: dict, skin_status: str, exit_reason: any,
                            language: str, temperature: float = 0.0, seed: int | None = None,
                            timeout: float | None = None, preempt_esi: int | None = None,
                            use_llm: bool = True, intake: dict | None = None) -> dict:
        """Synthesizes the complete patient encounter into the final ESI Level + SOAP note.

        ``temperature``, ``seed`` and ``timeout`` exist for experiments (M3 replay); the kiosk
        uses the defaults (temperature 0, no seed, SystemConfig.OLLAMA_FINAL_TIMEOUT).
        ``preempt_esi`` is the level found by the mid-interview emergency check; ``use_llm=False``
        skips the LLM (emergency fast path, so the nurse alert is not delayed). Every result
        passes through ``apply_safety_floor``.
        """
        transcript = " ".join([t["text"] for t in conversation_history if t.get("role") == "patient"])
        scenario = self.evaluate_scenario(transcript, visual_signs)

        exit_val = exit_reason.value if hasattr(exit_reason, "value") else str(exit_reason)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an Emergency Department Triage Officer operating under official ESI v4 standards.
Classify the patient into an ESI Level: 1 (Resuscitation), 2 (Emergent), 3 (Urgent), 4 (Less Urgent), or 5 (Non-urgent).

CLINICAL BOUNDARY CRITERIA:
- ESI Level 1: Immediate life-saving intervention required (cardiac arrest, airway compromise).
- ESI Level 2: High risk situation, acute chest pain with Levine's sign, severe pain rated 8-10/10 with acute distress, Altered Mental Status.
- ESI Level 3: Urgent condition needing 2 or more resources (e.g. lab test, IV meds, CT scan), moderate pain 4-7/10, but patient is stable, lucid, no acute danger.
- ESI Level 4: Requires 1 resource (e.g. simple x-ray or single medication).
- ESI Level 5: No resources needed (refill, exam only).

Write every text field in ENGLISH (translate Thai), at most 25 words per field.
Return ONLY a valid JSON object matching this schema:
{{
  "esi_level": <int 1-5>,
  "urgency": "<Resuscitation|Emergent|Urgent|Less Urgent|Non-urgent>",
  "scenario_type": "{scenario['scenario_type']}",
  "subjective": "<patient stated symptoms>",
  "objective": "<observed visual postures and skin assessment>",
  "assessment": "<clinical nursing impression>",
  "plan": "<immediate triage action plan>",
  "audit_mode": "LLM_INFERENCE_VALIDATED"
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
[CUMULATIVE PATIENT TRANSCRIPT]: "{transcript}"
[SUSTAINED PERCEPTION TELEMETRY]:
- Chest Pain (Levine's sign): {visual_signs.get('chest_clutching', False)}
- Abdominal Guarding: {visual_signs.get('abdominal_clutching', False)}
- Cranial Distress (Headache): {visual_signs.get('head_clutching', False)}
- Facial Skin Status (CNN): {skin_status}
[STRUCTURED INTAKE]: {self._intake_line(intake)}
[SCENARIO CROSS-VALIDATION]: {scenario['scenario_type']} ({scenario['clinical_flag']})
[ENCOUNTER EXIT REASON]: {exit_val}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": float(temperature), "num_predict": 200},
            "keep_alive": -1
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)
        request_timeout = timeout if timeout is not None else SystemConfig.OLLAMA_FINAL_TIMEOUT

        verdict = None
        fallback_reason = None
        if not use_llm:
            fallback_reason = "DETERMINISTIC_EMERGENCY_FAST_PATH"
        else:
            try:
                start_t = time.time()
                res = requests.post(self.endpoint, json=payload, timeout=request_timeout)
                if res.status_code != 200:
                    fallback_reason = f"DETERMINISTIC_FALLBACK_OLLAMA_HTTP_{res.status_code}"
                else:
                    out = json.loads(res.json().get("response", "{}"))
                    problem = self.verdict_problem(out)
                    if problem:
                        fallback_reason = "DETERMINISTIC_FALLBACK_INVALID_LLM_VERDICT"
                        logger.warning(f"LLM verdict rejected: {problem}")
                    else:
                        out["esi_level"] = int(float(out["esi_level"]))
                        out["scenario_details"] = scenario
                        out["inference_latency_sec"] = round(time.time() - start_t, 2)
                        verdict = out
            except requests.Timeout:
                fallback_reason = "DETERMINISTIC_FALLBACK_LLM_TIMEOUT"
            except requests.ConnectionError:
                fallback_reason = "DETERMINISTIC_FALLBACK_OLLAMA_UNREACHABLE"
            except (json.JSONDecodeError, ValueError):
                fallback_reason = "DETERMINISTIC_FALLBACK_JSON_PARSE_ERROR"
            except Exception as e:
                fallback_reason = f"DETERMINISTIC_FALLBACK_UNEXPECTED_{type(e).__name__.upper()}"

        rule_verdict = self._deterministic_verdict(transcript, visual_signs, skin_status, exit_val,
                                                   scenario, fallback_reason or "")
        if verdict is None:
            if fallback_reason != "DETERMINISTIC_EMERGENCY_FAST_PATH":
                logger.warning(f"Final synthesis fallback triggered: {fallback_reason}")
            verdict = rule_verdict
        else:
            # Fill any SOAP field the LLM left empty with the rule-based text.
            for key in ("urgency", "subjective", "objective", "assessment", "plan"):
                if not str(verdict.get(key) or "").strip():
                    verdict[key] = rule_verdict[key]
            verdict.setdefault("scenario_type", scenario["scenario_type"])
            verdict["audit_mode"] = "LLM_INFERENCE_VALIDATED"

        if intake:
            verdict["intake"] = intake
        return self.apply_safety_floor(verdict, visual_signs, skin_status, exit_val, scenario, preempt_esi,
                                       pain_score=(intake or {}).get("pain_score_0_10"))

    @staticmethod
    def _intake_line(intake: dict | None) -> str:
        if not intake:
            return "not collected"
        score = intake.get("pain_score_0_10")
        return (f"complaint='{intake.get('complaint') or 'not stated'}'; "
                f"pain score={'%s/10' % score if score is not None else intake.get('severity') or 'not stated'}; "
                f"onset='{intake.get('onset') or 'not stated'}'")

    # -------------------------------------------------------------------------
    # 7. VERDICT VALIDATION + SAFETY FLOOR
    # -------------------------------------------------------------------------
    REQUIRED_TEXT_FIELDS = ("urgency", "subjective", "objective", "assessment", "plan")
    URGENCY_BY_ESI = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-urgent"}

    @staticmethod
    def verdict_problem(out) -> str:
        """Return why an LLM verdict is unusable, or "" when it is acceptable."""
        if not isinstance(out, dict):
            return "not a JSON object"
        esi = out.get("esi_level")
        if isinstance(esi, bool):
            return "esi_level is not a number"
        try:
            esi_float = float(esi)
        except (TypeError, ValueError):
            return f"esi_level={esi!r} is not a number"
        if esi_float != int(esi_float):
            return f"esi_level={esi!r} is not a whole number"
        esi_int = int(esi_float)
        if not 1 <= esi_int <= 5:
            return f"esi_level={esi_int} outside 1-5"
        return ""

    @classmethod
    def apply_safety_floor(cls, verdict: dict, visual_signs: dict, skin_status: str, exit_val: str,
                           scenario: dict, preempt_esi=None, pain_score=None) -> dict:
        """Never let a verdict be LESS urgent than the hard signals the kiosk observed.

        A lower ESI number is more urgent. Each rule gives the highest ESI number allowed.
        """
        caps = []
        if exit_val == "EMERGENCY_INTERRUPT":
            caps.append((min(int(preempt_esi), 2) if preempt_esi in (1, 2) else 2,
                         "emergency detected during the interview"))
        if visual_signs.get("chest_clutching", False):
            caps.append((2, "sustained chest clutching (Levine's sign)"))
        if exit_val == "AMS_3_STRIKES":
            caps.append((2, "possible altered mental status (3 off-topic replies)"))
        skin = (skin_status or "").strip().upper().replace(" ", "_")
        if skin in ("RED_FLUSHING", "FLUSHING_(RED)", "PALLOR_(PALE)"):
            caps.append((3, f"abnormal skin ({skin_status})"))
        if scenario.get("scenario_type") == "SCENARIO_C_CLINICAL_CONFLICT":
            caps.append((3, "pain posture while the patient minimises symptoms"))
        if isinstance(pain_score, (int, float)) and pain_score >= 7:
            caps.append((3, f"patient-rated pain {pain_score}/10"))
        if not caps:
            return verdict
        limit = min(c[0] for c in caps)
        original = int(verdict["esi_level"])
        if original > limit:
            verdict["esi_before_safety_floor"] = original
            verdict["esi_level"] = limit
            reasons = "; ".join(r for c, r in caps if c == limit)
            verdict["urgency"] = f"{cls.URGENCY_BY_ESI[limit]} [SAFETY FLOOR: {reasons}]"
            verdict["safety_floor"] = reasons
            logger.warning(f"Safety floor: ESI {original} -> {limit} ({reasons})")
        return verdict

    def _deterministic_verdict(self, transcript: str, visual_signs: dict, skin_status: str, exit_val: str,
                               scenario: dict, fallback_reason: str) -> dict:
        """Rule-based ESI + SOAP note used when the LLM is unavailable or rejected."""
        # Critical keywords → ESI 1
        ESI1_KEYWORDS = [
            "หมดสติ", "ไม่รู้สึกตัว", "หยุดหายใจ", "ชักเกร็ง",
            "unconscious", "not breathing", "cardiac arrest",
            "seizure", "unresponsive"
        ]

        # Clearly minor keywords → ESI 5
        ESI5_KEYWORDS = [
            "ไข้หวัด", "น้ำมูก", "คัดจมูก", "ขอยา", "ต่ออายุยา",
            "cold", "runny nose", "refill", "prescription renewal",
            "minor cut", "small bruise"
        ]

        text_lower = transcript.lower()

        # --- Prerequisite flags ---
        no_visual_signs = not any([
            visual_signs.get("chest_clutching", False),
            visual_signs.get("abdominal_clutching", False),
            visual_signs.get("head_clutching", False)
        ])
        normalized_skin_status = skin_status.strip().upper().replace(" ", "_")
        normal_skin = normalized_skin_status == "NORMAL"
        no_conflict    = scenario["scenario_type"] != "SCENARIO_C_CLINICAL_CONFLICT"
        normal_exit    = exit_val not in ("AMS_3_STRIKES", "EMERGENCY_INTERRUPT", "EMERGENCY_ABORT",
                                           "HUMAN_NURSE_SUMMONED", "SYSTEM_FAULT")

        # --- Step 1: ESI 1 — check first, must never be missed ---
        is_esi1 = any(w in text_lower for w in ESI1_KEYWORDS)
        # AMS + chest simultaneously = cannot rule out cardiac collapse
        if exit_val == "AMS_3_STRIKES" and visual_signs.get("chest_clutching", False):
            is_esi1 = True

        # --- Step 2: Assign base ESI ---
        if is_esi1:
            esi    = 1
            urgency = "Resuscitation — Immediate Life-Saving Intervention Required"

        elif visual_signs.get("chest_clutching", False):
            esi    = 2
            urgency = "Emergent / High Risk"

        elif visual_signs.get("abdominal_clutching", False) or visual_signs.get("head_clutching", False):
            esi    = 3
            urgency = "Urgent (2+ Resources Anticipated)"

        elif exit_val == "AMS_3_STRIKES":
            # Cannot verify acuity → conservative
            esi    = 3
            urgency = "Urgent — Communication Barrier (Cannot Confirm Acuity)"

        elif no_visual_signs and normal_skin and no_conflict and normal_exit:
            # All clear conditions met → safe to assign ESI 4 or 5
            # English keywords on whole words, so "cold sweat" is not read as "cold".
            minor = any(w in text_lower for w in ESI5_KEYWORDS if not w.isascii()) or any(
                re.search(rf"\b{re.escape(w)}\b(?! sweat)", text_lower) for w in ESI5_KEYWORDS if w.isascii())
            if minor:
                esi    = 5
                urgency = "Non-urgent (No Resources Anticipated)"
            else:
                esi    = 4
                urgency = "Less Urgent (1 Resource Anticipated)"

        else:
            # LLM failed + unknown risk → conservative
            esi    = 3
            urgency = "Urgent (LLM Unavailable — Conservative Default)"

        # --- Step 3: Escalate if CONFLICT scenario (stoic patient) ---
        if scenario["scenario_type"] == "SCENARIO_C_CLINICAL_CONFLICT":
            if esi > 2:
                esi    -= 1
                urgency += " [ESCALATED: Stoic Patient Conflict]"

        # --- Step 4: Escalate if skin abnormal ---
        if normalized_skin_status in ("PALLOR_(PALE)", "FLUSHING_(RED)", "RED_FLUSHING"):
            if esi > 2:
                esi    -= 1
                urgency += f" [ESCALATED: Skin={skin_status}]"

        # --- FIX: dynamic assessment + plan based on actual signals ---
        if exit_val == "AMS_3_STRIKES":
            assessment = (
                "Patient repeatedly provided off-topic or non-clinical responses (3 strikes). "
                "Communication barrier or altered cognition cannot be excluded. "
                f"Visual posturing: chest={visual_signs.get('chest_clutching', False)}, "
                f"abdominal={visual_signs.get('abdominal_clutching', False)}, "
                f"head={visual_signs.get('head_clutching', False)}. "
                f"Skin: {skin_status}. Scenario: {scenario['scenario_type']}."
            )
            plan = (
                "Route to urgent assessment bay. Nurse to perform direct verbal assessment. "
                "Consider altered mental status workup. Do NOT discharge without human evaluation."
            )
        elif visual_signs.get("chest_clutching", False):
            assessment = (
                f"Sustained Levine's sign observed (chest guarding). "
                f"Skin status: {skin_status}. "
                f"Scenario cross-validation: {scenario['scenario_type']} — {scenario['clinical_flag']}. "
                "Cardiac or respiratory emergency cannot be ruled out at this stage."
            )
            plan = (
                "Route immediately to resuscitation bay. Initiate 12-lead ECG and IV access. "
                "Continuous SpO2 and BP monitoring. Notify on-call physician immediately."
            )
        elif visual_signs.get("abdominal_clutching", False):
            assessment = (
                f"Sustained abdominal guarding observed. "
                f"Verbal complaint: '{transcript[:120]}'. "
                f"Skin status: {skin_status}. "
                f"Scenario: {scenario['scenario_type']} — {scenario['clinical_flag']}."
            )
            plan = (
                "Route to acute care bay. IV access and basic labs (CBC, LFT, amylase) anticipated. "
                "NPO until surgical assessment complete. Monitor vitals every 15 minutes."
            )
        elif visual_signs.get("head_clutching", False):
            assessment = (
                f"Sustained cranial distress posturing observed. "
                f"Verbal complaint: '{transcript[:120]}'. "
                f"Skin status: {skin_status}. "
                f"Scenario: {scenario['scenario_type']} — {scenario['clinical_flag']}."
            )
            plan = (
                "Route to acute care bay. Neurological assessment required. "
                "Monitor for altered consciousness, photophobia, or focal deficits. "
                "CT head if clinically indicated."
            )
        else:
            assessment = (
                f"No critical posturing detected. Verbal complaint: '{transcript[:120]}'. "
                f"Skin status: {skin_status}. "
                f"Scenario: {scenario['scenario_type']}."
            )
            plan = (
                "Route to standard outpatient queue. Single resource anticipated. "
                "Reassess if condition worsens while waiting."
            )


        return {
            "esi_level": esi,
            "urgency": urgency,
            "scenario_type": scenario["scenario_type"],
            "subjective": transcript,
            "objective": f"Visual signs: {visual_signs}, Facial Skin Status (CNN): {skin_status}",
            "assessment": assessment,
            "plan": plan,
            "audit_mode": fallback_reason 
        }