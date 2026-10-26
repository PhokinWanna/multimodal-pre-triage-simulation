"""Kiosk orchestrator: runs the camera loop and the triage state machine.

The camera loop stays on the main thread (required by macOS for OpenCV windows),
while the encounter state machine runs in a background thread. Completed cases
are queued to the nurse-dashboard outbox.

Run from the repository root with: PYTHONPATH=src python -m triage_kiosk.main
"""

import logging
import os
import cv2
import time
import json
import queue
import threading
import uuid
import requests

from .config import SystemConfig, KioskState, ExitReason
from .consent import prepare_pdpa_refusal_handoff
from .vision.engine import VisionEngine
from .audio.engine import AudioEngine
from .triage.reasoning import ReasoningEngine

logger = logging.getLogger(__name__)


class ConveyorBeltWorker(threading.Thread):
    """Background Outbox Spooler (The Conveyor Belt)."""
    def __init__(self, spool_dir: str, endpoint: str):
        super().__init__(daemon=True)
        self.spool_dir = spool_dir
        self.endpoint = endpoint
        self.queue = queue.Queue()
        self.running = True
        os.makedirs(self.spool_dir, exist_ok=True)

    def enqueue_case(self, case_container: dict):
        case_id = case_container.get("case_id", f"CASE-{int(time.time())}")
        disk_path = os.path.join(self.spool_dir, f"{case_id}.json")

        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump(case_container, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to spool case to local disk: {e}")

        self.queue.put((f"Case {case_id}", disk_path, case_container))
        logger.info(f"[Conveyor Belt]: Case {case_id} queued for background transmission.")

    def enqueue_nurse_assistance(self, event: dict):
        """Persist and enqueue an anonymous, minimal nurse-assistance event."""
        event_id = uuid.uuid4().hex
        disk_path = os.path.join(self.spool_dir, f"nurse-assistance-{event_id}.json")

        try:
            with open(disk_path, "x", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to persist consent-refusal nurse handoff: {e}")
            raise

        self.queue.put(("Anonymous nurse-assistance event", disk_path, event))
        logger.info("Anonymous nurse-assistance event queued for dashboard delivery.")

    def run(self):
        while self.running:
            try:
                item_label, disk_path, payload = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            transmitted = False
            while not transmitted and self.running:
                try:
                    res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.HTTP_TIMEOUT)
                    if res.status_code in [200, 201]:
                        logger.info(f"[Conveyor Belt]: {item_label} delivered to Nurse Dashboard.")
                        transmitted = True
                        if os.path.exists(disk_path):
                            os.remove(disk_path)
                    else:
                        logger.warning(f"[Conveyor Belt]: {item_label}; dashboard status {res.status_code}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                        time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)
                except requests.exceptions.RequestException:
                    logger.warning(f"[Conveyor Belt]: Dashboard unreachable for {item_label}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                    time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)

            self.queue.task_done()


class SmartReceptionTriageKiosk:
    def __init__(self):
        logger.info("Initializing Smart Reception Triage Kiosk...")
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.reasoning = ReasoningEngine()

        self.last_heartbeat = time.time()

        # Arm the Conveyor Belt Outbox Worker
        self.conveyor_belt = ConveyorBeltWorker(
            spool_dir=SystemConfig.OUTBOX_SPOOL_DIR,
            endpoint=SystemConfig.NURSE_DASHBOARD_ENDPOINT
        )
        self.conveyor_belt.start()

        # Evidence folder setup
        self.evidence_dir = getattr(SystemConfig, "EVIDENCE_DIR", os.path.join(SystemConfig.BASE_DIR, "evidence"))
        os.makedirs(self.evidence_dir, exist_ok=True)

        self.current_state = KioskState.PREPARE_STANDBY
        self.active_language = "th"
        self.running = True
        self.current_strikes = 0

        self.latest_telemetry = {}
        self.telemetry_lock = threading.Lock()

        self.case_container = {}
        self._reset_case_container()

    def _reset_case_container(self):
        """Initializes a fresh clinical case container."""
        self.current_strikes = 0
        self.case_container = {
            "case_id": f"CASE-{int(time.time())}",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": self.active_language,
            "pdpa_consented": False,
            "exit_reason": None,
            "conversation": [],
            "cumulative_visual_signs": {
                "chest_clutching": False,
                "abdominal_clutching": False,
                "head_clutching": False
            },
            "facial_skin_status": "NORMAL",
            "visual_evidence_path": None,
            "triage_verdict": None
        }

    def _update_cumulative_visuals(self):
        """Updates sustained gestures and captures clean JPEG evidence without Base64 clutter."""
        with self.telemetry_lock:
            t = self.latest_telemetry
            
            # Use SUSTAINED gestures (>=3.0s) to eliminate the passing-hand latching bug!
            sustained = t.get("sustained_gestures", {})
            for key in ["chest_clutching", "abdominal_clutching", "head_clutching"]:
                if sustained.get(key, False):
                    self.case_container["cumulative_visual_signs"][key] = True

            self.case_container["facial_skin_status"] = t.get("skin_status", "NORMAL")

            # Save clean JPEG to disk
            if "clean_view" in t and self.case_container["visual_evidence_path"] is None:
                img_path = os.path.join(self.evidence_dir, f"{self.case_container['case_id']}.jpg")
                cv2.imwrite(img_path, t["clean_view"])
                self.case_container["visual_evidence_path"] = img_path
                logger.info(f"Captured JPEG Evidence: {img_path}")

    def _fsm_worker(self):
        """Background Agentic FSM Worker with Turn-by-Turn Telemetry Timers."""
        time.sleep(1.0)
        logger.info("Autonomous Reception Agent is active in background.")

        while self.running:
            try:
                # -------------------------------------------------------------
                # 1. State: PREPARE_STANDBY
                # -------------------------------------------------------------
                if self.current_state == KioskState.PREPARE_STANDBY:
                    if (time.time() - self.last_heartbeat) >= 240.0:
                        self.reasoning.warmup()
                        self.last_heartbeat = time.time()

                    with self.telemetry_lock:
                        patient_in_zone = self.latest_telemetry.get("patient_present", False)

                    if patient_in_zone:
                        logger.info("Patient in triage zone. Listening for speech or wake word...")
                        ambient_speech = self.audio.listen(language="auto")
                        triggered, detected_lang = self.audio.detect_wake_word(ambient_speech)

                        if triggered or len(ambient_speech) > 0:
                            logger.info("Patient engagement confirmed. Launching intake...")
                            self._reset_case_container()
                            self.active_language = detected_lang if triggered else "th"
                            self.case_container["language"] = self.active_language
                            self.current_state = KioskState.GREETING_LANG_SELECT

                # -------------------------------------------------------------
                # 2. State: GREETING_LANG_SELECT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.GREETING_LANG_SELECT:
                    self.audio.play_template("greeting", lang=self.active_language, block=True)
                    lang_reply = self.audio.listen(language=self.active_language)
                    logger.debug(f"[Language Preference]: \"{lang_reply}\"")

                    if any(w in lang_reply.lower() for w in ["english", "eng", "hello"]):
                        self.active_language = "en"
                    else:
                        self.active_language = "th"

                    self.case_container["language"] = self.active_language
                    self.current_state = KioskState.PDPA_CONSENT

                # -------------------------------------------------------------
                # 3. State: PDPA_CONSENT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.PDPA_CONSENT:
                    self.audio.play_template("pdpa_notice", lang=self.active_language, block=True)
                    consent_reply = self.audio.listen(language=self.active_language)

                    # Check for refusal
                    deny_tokens = ["ไม่", "ปฏิเสธ", "no", "deny", "disagree", "not consent"]
                    if any(t in consent_reply.lower() for t in deny_tokens):
                        logger.info("Patient refused PDPA consent. Transitioning to MANUAL_ROUTING...")
                        self.refusal_handoff = prepare_pdpa_refusal_handoff(self.case_container)
                        with self.telemetry_lock:
                            self.latest_telemetry = {}
                        self.current_state = KioskState.MANUAL_ROUTING
                    else:
                        self.case_container["conversation"].append({"role": "pdpa_response", "text": consent_reply})
                        self.case_container["pdpa_consented"] = True
                        self.current_state = KioskState.SYMPTOM_INTERVIEW

                # -------------------------------------------------------------
                # 3b. State: MANUAL_ROUTING
                # -------------------------------------------------------------
                elif self.current_state == KioskState.MANUAL_ROUTING:
                    self.audio.play_template("pdpa_denied", lang=self.active_language, block=True)
                    try:
                        self.conveyor_belt.enqueue_nurse_assistance(self.refusal_handoff)
                        del self.refusal_handoff
                    except OSError as e:
                        logger.error(
                            "Dashboard handoff could not be queued. "
                            f"Follow the audio instruction to contact the nurse directly: {e}"
                        )
                    time.sleep(2.0)
                    logger.info("Patient routed to human desk. Resetting to PREPARE_STANDBY.")
                    self.current_state = KioskState.PREPARE_STANDBY

                # -------------------------------------------------------------
                # 4. State: SYMPTOM_INTERVIEW (Agentic Action Loop + Timers)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.SYMPTOM_INTERVIEW:
                    with self.telemetry_lock:
                        dom_gesture = self.latest_telemetry.get("dominant_current_gesture", "NONE")

                    proactive_prompt = self.reasoning.get_proactive_visual_prompt(dom_gesture, lang=self.active_language)

                    if proactive_prompt:
                        logger.info(f"[Proactive Visual Cue]: Triggering inquiry for {dom_gesture}")
                        self.audio.play_template("inquiry", lang=self.active_language, block=True)
                        self.case_container["conversation"].append({"role": "kiosk_proactive", "text": proactive_prompt})
                    else:
                        self.audio.play_template("inquiry", lang=self.active_language, block=True)

                    interview_active = True
                    consecutive_pauses = 0

                    while interview_active and self.running:
                        turn_start_time = time.time()  # ⏱️ Start Turn Timer

                        # 1. Capture speech (STT)
                        t_stt_start = time.time()
                        chunk = self.audio.listen(language=self.active_language)
                        stt_duration = time.time() - t_stt_start

                        self._update_cumulative_visuals()

                        # Silence Handling
                        if len(chunk) == 0:
                            consecutive_pauses += 1
                            if consecutive_pauses == 1:
                                self.audio.play_template("still_listening", lang=self.active_language, block=True)
                                continue
                            elif consecutive_pauses == 2:
                                self.audio.play_template("anything_else", lang=self.active_language, block=True)
                                continue
                            else:
                                logger.info("Extended silence verified. Finalizing encounter.")
                                self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                                interview_active = False
                                break

                        consecutive_pauses = 0
                        self.case_container["conversation"].append({"role": "patient", "text": chunk})

                        # Emergency Preemption Check
                        is_preempted, preempt_esi = self.reasoning.check_acute_preemption(
                            chunk,
                            self.case_container["cumulative_visual_signs"]
                        )
                        if is_preempted:
                            logger.warning(f"ACUTE EMERGENCY PREEMPTION: ESI Level {preempt_esi}!")
                            self.case_container["exit_reason"] = ExitReason.EMERGENCY_INTERRUPT.value
                            interview_active = False
                            break

                        # Stopping Intent Check ("แค่นี้ครับ" / "หมดแล้ว")
                        if self.reasoning.is_stopping_phrase(chunk, lang=self.active_language):
                            logger.info("Patient explicitly requested completion.")
                            self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                            interview_active = False
                            break

                        # 2. Agent Decision (LLM Reasoning)
                        with self.telemetry_lock:
                            v_telemetry = self.latest_telemetry

                        action_frame = self.reasoning.dispatch_agent_turn(
                            utterance=chunk,
                            history=self.case_container["conversation"],
                            visual_telemetry=v_telemetry,
                            current_strikes=self.current_strikes,
                            active_lang=self.active_language
                        )

                        action = action_frame.get("action", "CONTINUE_TRIAGE")
                        spoken_resp = action_frame.get("spoken_response", "")
                        llm_duration = action_frame.get("llm_latency_sec", 0.0)
                        self.current_strikes = action_frame.get("strikes", self.current_strikes)

                        logger.info(f"[Agent Decision]: Action={action} | Spoken=\"{spoken_resp}\"")

                        # 3. Audio Playback (TTS)
                        t_tts_start = time.time()

                        # ACTION: TRIAGE COMPLETE (Onset + Severity gathered -> Stop asking!)
                        if action == "TRIAGE_COMPLETE":
                            logger.info("All necessary clinical data gathered. Concluding interview.")
                            self.audio.play_template("anything_else", lang=self.active_language, block=True)
                            final_reply = self.audio.listen(language=self.active_language)
                            
                            if len(final_reply) > 0 and not self.reasoning.is_stopping_phrase(final_reply, lang=self.active_language):
                                self.case_container["conversation"].append({"role": "patient", "text": final_reply})

                            self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                            interview_active = False
                            break

                        # ACTION: Cancel and Exit (Instant Reset - Bypasses slow SOAP generation)
                        elif action == "CANCEL_AND_EXIT":
                            logger.info("AGENT COMMAND: Patient requested early cancellation.")
                            self.audio.play_template("wrapup", lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = "USER_CANCELLED"
                            interview_active = False
                            break

                        elif action == "SWITCH_LANGUAGE":
                            new_lang = action_frame.get("target_language", "en")
                            logger.info(f"AGENT COMMAND: Switching language from {self.active_language} -> {new_lang}")
                            self.active_language = new_lang
                            self.case_container["language"] = self.active_language
                            self.audio.play_template("inquiry", lang=self.active_language, block=True)
                            self.case_container["conversation"].append({"role": "kiosk_action", "text": spoken_resp})

                        elif action == "CALL_HUMAN_NURSE":
                            logger.warning("AGENT COMMAND: Summoning human triage staff.")
                            self.audio.play_template("ams_alert", lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = "HUMAN_NURSE_SUMMONED"
                            interview_active = False
                            break

                        elif action == "EMERGENCY_ABORT":
                            logger.warning("AGENT COMMAND: High-Acuity Emergency Abort triggered!")
                            self.audio.play_template("emergency_alert", lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = ExitReason.EMERGENCY_INTERRUPT.value
                            interview_active = False
                            break

                        elif action == "TRIGGER_AMS":
                            logger.warning("3-Strike Rule Tripped: Altered Mental Status (AMS).")
                            self.audio.play_template("ams_alert", lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = ExitReason.AMS_3_STRIKES.value
                            interview_active = False
                            break

                        else:
                            if action_frame.get("is_off_topic", False):
                                self.audio.speak_off_topic(lang=self.active_language, block=True)
                            else:
                                self.audio.play_template("inquiry", lang=self.active_language, block=True)
                            self.case_container["conversation"].append({"role": "kiosk_agent", "text": spoken_resp})

                        tts_duration = time.time() - t_tts_start
                        total_turnaround = time.time() - turn_start_time

                        # ⏱️ Real-Time Telemetry Benchmark Printout
                        logger.info(f"[TELEMETRY]: STT: {stt_duration:.2f}s | LLM: {llm_duration:.2f}s | TTS: {tts_duration:.2f}s | Turnaround: {total_turnaround:.2f}s")

                    # If cancelled, bypass heavy SOAP evaluation and reset immediately
                    if self.case_container.get("exit_reason") == "USER_CANCELLED":
                        self.current_state = KioskState.DISPATCH_AND_RESET
                    else:
                        self.current_state = KioskState.FINALIZING_CASE

                # -------------------------------------------------------------
                # 5. State: FINALIZING_CASE (Holistic SOAP Synthesis)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.FINALIZING_CASE:
                    raw_exit = self.case_container.get("exit_reason", ExitReason.NORMAL_COMPLETE.value)
                    exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)

                    logger.info(f"Packaging case [{self.case_container['case_id']}] (Exit: {exit_str})...")
                    self._update_cumulative_visuals()

                    verdict = self.reasoning.evaluate_final_case(
                        conversation_history=self.case_container["conversation"],
                        visual_signs=self.case_container["cumulative_visual_signs"],
                        skin_status=self.case_container["facial_skin_status"],
                        exit_reason=raw_exit,
                        language=self.active_language
                    )
                    self.case_container["triage_verdict"] = verdict
                    logger.info(f"[VERDICT]: ESI Level {verdict.get('esi_level')} ({verdict.get('urgency')})")
                    self.current_state = KioskState.DISPATCH_AND_RESET

                # -------------------------------------------------------------
                # 6. State: DISPATCH_AND_RESET (Conveyor Belt Transport)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.DISPATCH_AND_RESET:
                    raw_exit = self.case_container.get("exit_reason", ExitReason.NORMAL_COMPLETE.value)
                    exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)

                    if exit_str == ExitReason.EMERGENCY_INTERRUPT.value:
                        self.audio.play_template("emergency_alert", lang=self.active_language, block=True)
                    elif exit_str == ExitReason.AMS_3_STRIKES.value:
                        self.audio.play_template("ams_alert", lang=self.active_language, block=True)
                    elif exit_str not in ["USER_CANCELLED", "HUMAN_NURSE_SUMMONED"]:
                        self.audio.play_template("wrapup", lang=self.active_language, block=True)

                    # Enqueue to background Conveyor Belt (<1ms)
                    self.conveyor_belt.enqueue_case(self.case_container)
                    logger.info("Case finalized. Kiosk returning to PREPARE_STANDBY.")
                    self.current_state = KioskState.PREPARE_STANDBY

                time.sleep(0.05)

            except Exception as e:
                logger.error(f"FSM Encounter Exception: {e}")
                time.sleep(1.0)

    def run(self):
        """Main-Thread GUI Capture Loop (macOS Cocoa Compliant)."""
        cap = cv2.VideoCapture(SystemConfig.CAMERA_INDEX)
        if not cap.isOpened():
            cap.release()
            self.running = False
            self.conveyor_belt.running = False
            self.vision.release()
            self.audio.release()
            raise RuntimeError(
                f"Could not open camera device {SystemConfig.CAMERA_INDEX}."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SystemConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SystemConfig.FRAME_HEIGHT)

        fsm_thread = threading.Thread(target=self._fsm_worker, daemon=True)
        fsm_thread.start()
        logger.info("Camera capture active on Main Thread.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                telemetry = self.vision.process_frame(frame)
                with self.telemetry_lock:
                    self.latest_telemetry = telemetry

                if SystemConfig.SHOW_DUAL_WINDOWS:
                    cv2.imshow("Kiosk - Clean Patient View", telemetry["clean_view"])
                    cv2.imshow("Kiosk - Diagnostic Skeleton View", telemetry["skeleton_view"])
                else:
                    cv2.imshow("Smart Triage Kiosk", telemetry["clean_view"])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        except KeyboardInterrupt:
            logger.info("Powering down systems gracefully...")
        finally:
            self.running = False
            self.conveyor_belt.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.vision.release()
            self.audio.release()
            logger.info("Kiosk successfully and safely offline.")


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    kiosk = SmartReceptionTriageKiosk()
    kiosk.run()