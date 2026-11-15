"""Final-verdict prompts for the local LLM.

* ``v1`` - the prompt used up to round A (five one-line level definitions, raw camera flags).
* ``v2`` - round B: the ESI decision steps in order (A: life-saving now? B: high risk? C: count
  the resources), an explicit ESI 5 definition, camera rules, and the step answers in the output.
* ``v3`` - v2 plus a few short worked examples.

Which one the kiosk uses is chosen on DEVELOPMENT scenarios only
(scripts/experiments/m3/prompt_dev_eval.py). The step examples are the general criteria of the
ESI handbook (airway/breathing/circulation/consciousness for step A; the handbook's high-risk
list for step B); the worked examples of v3 appear in no test or development scenario.
"""

from __future__ import annotations

from typing import Optional

PROMPT_VERSIONS = ("v1", "v2", "v3")
NUM_PREDICT = {"v1": 200, "v2": 280, "v3": 280}
NO_WORDS = "(the patient did not describe any symptom)"

_HEAD = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
_USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
_ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

V1_SYSTEM = """You are an Emergency Department Triage Officer operating under official ESI v4 standards.
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
  "scenario_type": "{scenario_type}",
  "subjective": "<patient stated symptoms>",
  "objective": "<observed visual postures and skin assessment>",
  "assessment": "<clinical nursing impression>",
  "plan": "<immediate triage action plan>",
  "audit_mode": "LLM_INFERENCE_VALIDATED"
}}
"""

V2_SYSTEM = """You are a triage nurse using the Emergency Severity Index (ESI). Decide the ESI level of ONE patient.
Answer the steps IN ORDER and stop at the first YES.

STEP A - Does the patient need a life-saving action RIGHT NOW?
  YES examples: not breathing or severe trouble breathing; airway swelling or blocked; no pulse, or signs of shock (pale, cold, sweaty, about to faint); heavy bleeding that will not stop; unresponsive, or suddenly very drowsy or confused.
  YES -> ESI 1.
STEP B - Is this a high-risk situation that must not wait?
  YES examples: chest pain or pressure that could be from the heart; signs of a stroke (sudden weakness on one side, slurred speech, loss of vision); new confusion or unusual sleepiness; severe pain or distress, 7 to 10 out of 10; a possible problem in pregnancy with pain or bleeding; fever in a patient with weak immunity; risk of self-harm.
  YES -> ESI 2.
STEP C - Otherwise count the RESOURCES this patient will probably need.
  Resources: blood or urine tests, ECG, X-ray, CT or ultrasound, IV fluids, IV or injected medicine, breathing treatment, specialist consult, stitches or another procedure.
  NOT resources: examination, tablets, a prescription or refill, simple dressing, advice, a certificate.
  two or more resources -> ESI 3.   exactly one -> ESI 4.   none -> ESI 5.

ESI 5 is the right answer for a minor problem that needs no test and no procedure. Do not raise it "to be safe".

CAMERA RULES
- Use the patient's words first. The camera can make a case MORE urgent, never less urgent.
- "Combined visual risk: HIGH", or a hand held on the chest, is a high-risk sign: answer STEP B yes even when the patient says it is nothing.
- MODERATE or LOW visual risk, a flushed face alone, or a hand on the head or abdomen alone does NOT change the level. Decide from the words.
- "not assessed" means no information.

Write the text fields in ENGLISH (translate Thai), at most 15 words each. Do not name a disease.
Return ONLY this JSON object:
{"step_a": "<yes or no>", "step_b": "<yes or no>", "resources": <0, 1 or 2 where 2 means two or more>, "esi_level": <1-5>, "urgency": "<Resuscitation|Emergent|Urgent|Less Urgent|Non-urgent>", "subjective": "<what the patient said>", "objective": "<camera findings>", "assessment": "<nursing impression>", "plan": "<immediate triage action>"}
"""

V3_EXAMPLES = """
EXAMPLES (only the first four fields are shown)
Patient: "I was hit by a car. My leg is bleeding heavily, I feel cold and I am about to pass out." Camera: nothing.
-> {"step_a": "yes", "step_b": "no", "resources": 2, "esi_level": 1}
Patient: "ตาขวามองไม่เห็นทันทีเมื่อสิบนาทีก่อน ไม่เจ็บค่ะ" Camera: nothing.
-> {"step_a": "no", "step_b": "yes", "resources": 2, "esi_level": 2}
Patient: "รู้สึกเหนื่อยๆ ไม่มีแรง ไม่ได้เจ็บตรงไหนครับ" Camera: hand held on the chest, combined visual risk HIGH.
-> {"step_a": "no", "step_b": "yes", "resources": 2, "esi_level": 2}
Patient: "ปวดเอวด้านซ้ายร้าวลงขาหนีบ ปัสสาวะเป็นเลือด ปวดประมาณห้าเต็มสิบครับ" Camera: nothing.
-> {"step_a": "no", "step_b": "no", "resources": 2, "esi_level": 3}
Patient: "I have had a cough for two weeks. No fever and I am not short of breath." Camera: nothing.
-> {"step_a": "no", "step_b": "no", "resources": 1, "esi_level": 4}
Patient: "ปวดหลังเล็กน้อยหลังยกของเมื่อวาน เดินได้ปกติ อยากได้ยาทาครับ" Camera: face flushed, combined visual risk MODERATE.
-> {"step_a": "no", "step_b": "no", "resources": 0, "esi_level": 5}
Patient: "ขอใบรับรองแพทย์ไปสมัครงานค่ะ สบายดี" Camera: nothing.
-> {"step_a": "no", "step_b": "no", "resources": 0, "esi_level": 5}
"""


def camera_block_v1(signs: dict, skin_status: str) -> str:
    return ("[SUSTAINED PERCEPTION TELEMETRY]:\n"
            f"- Chest Pain (Levine's sign): {signs.get('chest_clutching', False)}\n"
            f"- Abdominal Guarding: {signs.get('abdominal_clutching', False)}\n"
            f"- Cranial Distress (Headache): {signs.get('head_clutching', False)}\n"
            f"- Facial Skin Status (CNN): {skin_status}")


def camera_block_v2(posture_text: str, skin_text: str, visual: Optional[dict]) -> str:
    """``visual`` = fusion.visual_score(...) in weighted mode, None when the two parts stay independent."""
    lines = ["[CAMERA]", f"- Posture: {posture_text}", f"- Facial skin: {skin_text}"]
    if visual is not None:
        if visual["score"] is None:
            lines.append("- Combined visual risk: not assessed")
        else:
            w = visual["weights"]
            lines.append(f"- Combined visual risk: {visual['band']} (score {visual['score']:.2f} of 1.00; "
                         f"posture weight {round(w['posture'] * 100)}%, skin weight {round(w['skin'] * 100)}%)")
    return "\n".join(lines)


def build_final_prompt(version: str, transcript: str, camera_block: str, intake_line: str,
                       scenario: dict, exit_val: str) -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version: {version}")
    words = transcript.strip()
    if version == "v1":
        return (_HEAD + V1_SYSTEM.format(scenario_type=scenario["scenario_type"]) + _USER
                + f'[CUMULATIVE PATIENT TRANSCRIPT]: "{words}"\n' + camera_block + "\n"
                + f"[STRUCTURED INTAKE]: {intake_line}\n"
                + f"[SCENARIO CROSS-VALIDATION]: {scenario['scenario_type']} ({scenario['clinical_flag']})\n"
                + f"[ENCOUNTER EXIT REASON]: {exit_val}\n" + _ASSISTANT)
    system = V2_SYSTEM + (V3_EXAMPLES if version == "v3" else "")
    patient = f'"{words}"' if words else NO_WORDS
    user = f"[PATIENT'S WORDS]: {patient}\n"
    if intake_line and intake_line != "not collected":
        user += f"[INTAKE ANSWERS]: {intake_line}\n"
    user += camera_block + "\n"
    if exit_val not in ("NORMAL_COMPLETE", ""):
        user += f"[INTERVIEW ENDED BY]: {exit_val}\n"
    return _HEAD + system + _USER + user + _ASSISTANT


def esi_from_steps(out: dict) -> Optional[int]:
    """ESI level implied by the model's own step answers (v2/v3), or None if they are unusable."""
    def yes_no(v):
        s = str(v).strip().lower()
        return True if s in ("yes", "true", "y") else False if s in ("no", "false", "n") else None

    a, b = yes_no(out.get("step_a")), yes_no(out.get("step_b"))
    if a is None or b is None:
        return None
    if a:
        return 1
    if b:
        return 2
    try:
        r = int(float(out.get("resources")))
    except (TypeError, ValueError):
        return None
    if r < 0:
        return None
    return 3 if r >= 2 else 4 if r == 1 else 5
