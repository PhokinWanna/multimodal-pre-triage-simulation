"""Reasoning engine: LLM-driven symptom interview and ESI scoring.

Talks to a local Llama 3.2 model through Ollama to choose the kiosk's next
action each turn and to propose a final ESI level, with deterministic
rule-based fallbacks when the LLM is unavailable or times out.
"""

import logging
import json
import time
import requests
from ..config import SystemConfig

logger = logging.getLogger(__name__)

class ReasoningEngine:
    def __init__(self):
        self.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"
        self.warmup()

    def warmup(self):
        """Pre-warms Llama 3.2 in VRAM on boot with a 15-second cold-load runway."""
        logger.info("Pre-warming Llama 3.2 in VRAM (Pre-Flight Test)...")
        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": "ok",
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": -1
        }
        try:
            # 15s timeout specifically for cold SSD-to-RAM loading
            requests.post(self.endpoint, json=payload, timeout=15.0)
            logger.info("Llama 3.2 is hot and pinned in VRAM.")
        except Exception as e:
            logger.warning(f"Warmup Notice: Ollama not reachable on boot: {e}")

    # -------------------------------------------------------------------------
    # 1. STOPPING INTENT DETECTOR
    # -------------------------------------------------------------------------
    @staticmethod
    def is_stopping_phrase(utterance: str, lang: str = "th") -> bool:
        """Detects if patient indicates they are done explaining."""
        text = utterance.strip().lower()
        if len(text) == 0:
            return False

        phrases = getattr(SystemConfig, "STOP_PHRASES_TH", ["หมดแล้ว", "แค่นี้", "ไม่มีแล้ว", "พอแค่นี้", "พอแล้ว"]) if lang == "th" else \
                  getattr(SystemConfig, "STOP_PHRASES_EN", ["that's all", "thats all", "nothing else", "no more", "done", "finished", "enough"])
        return any(p in text for p in phrases)

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
        if any(w in text for w in ["พอแค่นี้", "ไม่เอาแล้ว", "ยกเลิก", "enough", "stop", "cancel", "going out", "see the doctor"]):
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
            "model": SystemConfig.OLLAMA_MODEL,
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
            res = requests.post(self.endpoint, json=payload, timeout=getattr(SystemConfig, "OLLAMA_TIMEOUT", 8.0))
            llm_duration = time.time() - t0

            if res.status_code == 200:
                raw_response = res.json().get("response", "").strip()

                # JSON Auto-Repair Guardrail
                if not raw_response.endswith("}"):
                    if '"' in raw_response and raw_response.count('"') % 2 != 0:
                        raw_response += '"'
                    raw_response += "}"

                frame = json.loads(raw_response)
                frame["llm_latency_sec"] = round(llm_duration, 2)

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
    def evaluate_final_case(self, conversation_history: list, visual_signs: dict, skin_status: str, exit_reason: any, language: str) -> dict:
        """Synthesizes the complete patient encounter into the final ESI Level + SOAP note."""
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
[SCENARIO CROSS-VALIDATION]: {scenario['scenario_type']} ({scenario['clinical_flag']})
[ENCOUNTER EXIT REASON]: {exit_val}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 256},
            "keep_alive": -1
        }

        try:
            start_t = time.time()
            res = requests.post(self.endpoint, json=payload, timeout=getattr(SystemConfig, "OLLAMA_TIMEOUT", 10.0))
            if res.status_code == 200:
                out = json.loads(res.json().get("response", "{}"))
                out["scenario_details"] = scenario
                out["inference_latency_sec"] = round(time.time() - start_t, 2)
                return out
        except requests.Timeout:
            fallback_reason = "DETERMINISTIC_FALLBACK_LLM_TIMEOUT"
        except requests.ConnectionError:
            fallback_reason = "DETERMINISTIC_FALLBACK_OLLAMA_UNREACHABLE"
        except json.JSONDecodeError:
            fallback_reason = "DETERMINISTIC_FALLBACK_JSON_PARSE_ERROR"
        except Exception as e:
            fallback_reason = f"DETERMINISTIC_FALLBACK_UNEXPECTED_{type(e).__name__.upper()}"

        logger.warning(f"Final synthesis fallback triggered: {fallback_reason}")

        # # Deterministic Fallback
        # if exit_val == "AMS_3_STRIKES" or visual_signs.get("chest_clutching", False):
        #     esi = 2
        #     urgency = "Emergent / High Risk"
        # elif visual_signs.get("abdominal_clutching", False) or visual_signs.get("head_clutching", False):
        #     esi = 3
        #     urgency = "Urgent (2+ Resources Anticipated)"
        # else:
        #     esi = 4
        #     urgency = "Less Urgent"

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
        normal_exit    = exit_val not in ("AMS_3_STRIKES", "EMERGENCY_ABORT")

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
            if any(w in text_lower for w in ESI5_KEYWORDS):
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