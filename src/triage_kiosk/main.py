"""Kiosk orchestrator: runs the camera loop and the triage state machine.

The camera loop stays on the main thread (required by macOS for OpenCV windows),
while the encounter state machine runs in a background thread. Completed cases
are queued to the nurse-dashboard outbox.

Run from the repository root with: PYTHONPATH=src python -m triage_kiosk.main
"""

import faulthandler
import logging
import os
import sys
import cv2
import time
import threading

from .config import SystemConfig, KioskState, ExitReason
from .platform_support import configure_console_utf8, open_camera
from .consent import kiosk_fault_event, prepare_pdpa_refusal_handoff
from .intent import (NO, UNCLEAR, YES, choose_language, classify_consent, is_cancel_request,
                     is_ready_to_finish, requested_language)
from .triage.intake import StructuredIntake
from .tts_privacy import safe_off_topic_text
from .outbox import ConveyorBeltWorker
from .vision.engine import VisionEngine
from .audio.engine import AudioEngine
from .triage.reasoning import ReasoningEngine

logger = logging.getLogger(__name__)


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
        self.last_fault_report = 0.0
        self.refusal_handoff = None
        self.cooldown_until = 0.0

        self.case_container = {}
        self._reset_case_container()

    def _reset_case_container(self):
        """Initializes a fresh clinical case container."""
        self.current_strikes = 0
        self.alert_played = False
        self.wrapup_played = False
        self.fault_finalize_tried = False
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
            "facial_skin_status": "UNKNOWN",
            "visual_evidence_path": None,
            "intake": None,
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

            self.case_container["facial_skin_status"] = t.get("skin_status", "UNKNOWN")

            # Save clean JPEG to disk
            if "clean_view" in t and self.case_container["visual_evidence_path"] is None:
                img_path = os.path.join(self.evidence_dir, f"{self.case_container['case_id']}.jpg")
                cv2.imwrite(img_path, t["clean_view"])
                self.case_container["visual_evidence_path"] = img_path
                logger.info(f"Captured JPEG Evidence: {img_path}")

    # ------------------------------------------------------------------
    # Speech helpers: what is logged is exactly what the patient heard
    # ------------------------------------------------------------------
    def _say(self, text: str, fallback_template: str, role: str = "kiosk_agent", log: bool = True) -> str:
        """Speak ``text`` with the offline voice; if no voice is available, play the template.

        The conversation log records the sentence that was actually played, so the
        nurse dashboard never shows a question the patient did not hear.
        """
        lang = self.active_language
        if text and self.audio.speak_text(text, lang=lang, block=True):
            spoken = text
        else:
            self.audio.play_template(fallback_template, lang=lang, block=True)
            spoken = SystemConfig.SCRIPT_TEXTS.get(lang, {}).get(fallback_template, "")
        if log and spoken:
            self.case_container["conversation"].append({"role": role, "text": spoken})
        return spoken

    def _play_logged(self, template: str, role: str = "kiosk_agent") -> None:
        self.audio.play_template(template, lang=self.active_language, block=True)
        text = SystemConfig.SCRIPT_TEXTS.get(self.active_language, {}).get(template, "")
        if text:
            self.case_container["conversation"].append({"role": role, "text": text})

    def _wait_audio_idle(self, max_seconds: float = 30.0) -> None:
        """Wait until a non-blocking prompt (e.g. the wrap-up) has finished playing."""
        time.sleep(0.2)
        end = time.time() + max_seconds
        while getattr(self.audio, "is_speaking", False) and time.time() < end:
            time.sleep(0.1)

    def _alert_now(self, template: str) -> None:
        """Play an alert immediately (before any slow step) and only once per case."""
        if not self.alert_played:
            self.audio.play_template(template, lang=self.active_language, block=True)
            self.alert_played = True

    def _emergency_stop(self, preempt_esi: int, reason: str) -> None:
        logger.warning(f"ACUTE EMERGENCY ({reason}): ESI Level {preempt_esi}!")
        self.case_container["exit_reason"] = ExitReason.EMERGENCY_INTERRUPT.value
        self.case_container["preempt_esi"] = preempt_esi
        # The alert is played in DISPATCH right after the case is queued (the emergency
        # fast path takes milliseconds), so the dashboard alert and the voice alert coincide.

    # ------------------------------------------------------------------
    # Fault handling
    # ------------------------------------------------------------------
    def _handle_fault(self, error: Exception) -> None:
        """Called after FSM_MAX_CONSECUTIVE_ERRORS failures in a row.

        Tells staff (anonymous dashboard event + alert sound when possible), keeps an
        in-progress consented case by finalising it as SYSTEM_FAULT, otherwise resets.
        """
        state = self.current_state
        logger.critical(f"KIOSK FAULT in state {state.value}: {error}. Alerting staff.")
        try:
            self.audio.play_template("ams_alert", lang=self.active_language, block=True)
        except Exception:  # noqa: BLE001  (the speaker may be the broken part)
            pass
        now = time.time()
        if now - self.last_fault_report >= 600:
            try:
                self.conveyor_belt.enqueue_nurse_assistance(kiosk_fault_event())
                self.last_fault_report = now
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not queue the kiosk-fault event: {e}")

        if (state == KioskState.SYMPTOM_INTERVIEW and self.case_container.get("pdpa_consented")
                and not self.fault_finalize_tried):
            self.fault_finalize_tried = True
            self.case_container["exit_reason"] = ExitReason.SYSTEM_FAULT.value
            self.current_state = KioskState.FINALIZING_CASE
        else:
            self.current_state = KioskState.PREPARE_STANDBY
            time.sleep(SystemConfig.FSM_FAULT_BACKOFF_SECONDS)

    def _fsm_worker(self):
        """Background FSM worker: one patient encounter at a time."""
        time.sleep(1.0)
        logger.info("Autonomous Reception Agent is active in background.")
        consecutive_errors = 0

        while self.running:
            try:
                self._fsm_step()
                consecutive_errors = 0
                time.sleep(0.05)
            except Exception as e:  # noqa: BLE001
                consecutive_errors += 1
                logger.error(f"FSM Encounter Exception ({consecutive_errors}/"
                             f"{SystemConfig.FSM_MAX_CONSECUTIVE_ERRORS}): {e}")
                if consecutive_errors >= SystemConfig.FSM_MAX_CONSECUTIVE_ERRORS:
                    consecutive_errors = 0
                    try:
                        self._handle_fault(e)
                    except Exception as e2:  # noqa: BLE001
                        logger.critical(f"Fault handler failed: {e2}")
                        self.current_state = KioskState.PREPARE_STANDBY
                else:
                    time.sleep(1.0)

    def _fsm_step(self):
        # -------------------------------------------------------------
        # 1. PREPARE_STANDBY
        # -------------------------------------------------------------
        if self.current_state == KioskState.PREPARE_STANDBY:
            if (time.time() - self.last_heartbeat) >= 240.0:
                self.reasoning.warmup()
                self.last_heartbeat = time.time()

            with self.telemetry_lock:
                patient_in_zone = self.latest_telemetry.get("patient_present", False)

            if not patient_in_zone:
                self.cooldown_until = 0.0          # the last person left: ready for the next one
            elif time.time() < getattr(self, "cooldown_until", 0.0):
                time.sleep(0.2)
            else:
                logger.info("Patient in triage zone. Listening for speech or wake word...")
                ambient_speech = self.audio.listen(language="auto")
                triggered, detected_lang = self.audio.detect_wake_word(ambient_speech)

                if triggered or len(ambient_speech) > 0:
                    logger.info("Patient engagement confirmed. Launching intake...")
                    self._reset_case_container()
                    spoken_lang = getattr(self.audio, "last_language", (None, 0.0))[0]
                    self.active_language = detected_lang if triggered else (spoken_lang or "th")
                    self.case_container["language"] = self.active_language
                    self.current_state = KioskState.GREETING_LANG_SELECT

        # -------------------------------------------------------------
        # 2. GREETING_LANG_SELECT
        # -------------------------------------------------------------
        elif self.current_state == KioskState.GREETING_LANG_SELECT:
            self.audio.calibrate()          # room noise right now, before the kiosk talks
            self.audio.play_template("greeting", lang=self.active_language, block=True)
            # "auto" = Thai or English, whichever the person speaks (a forced-Thai transcript of
            # "English please" comes out in Thai letters and was missed before).
            lang_reply = self.audio.listen(language="auto")
            detected, conf = getattr(self.audio, "last_language", (None, 0.0))
            new_lang = choose_language(lang_reply, detected, conf, self.active_language)
            logger.info(f"[Language]: reply in {detected} ({conf:.2f}) -> using {new_lang}")
            self.active_language = new_lang

            self.case_container["language"] = self.active_language
            self.current_state = KioskState.PDPA_CONSENT

        # -------------------------------------------------------------
        # 3. PDPA_CONSENT: explicit "yes" required; silence/unclear is NOT consent
        # -------------------------------------------------------------
        elif self.current_state == KioskState.PDPA_CONSENT:
            decision, consent_reply = UNCLEAR, ""
            attempts, switched, replay_notice = 0, False, True
            while attempts < SystemConfig.PDPA_MAX_ASKS:
                if replay_notice:
                    self.audio.play_template("pdpa_notice", lang=self.active_language, block=True)
                    replay_notice = False
                else:
                    reask = SystemConfig.SPOKEN_PROMPTS[self.active_language]["pdpa_reask"]
                    if not self.audio.speak_text(reask, lang=self.active_language, block=True):
                        self.audio.play_template("pdpa_notice", lang=self.active_language, block=True)
                consent_reply = self.audio.listen(language="auto")
                detected, conf = getattr(self.audio, "last_language", (None, 0.0))
                decision = classify_consent(consent_reply)

                # The person may still change language here ("English please", or simply
                # answering in the other language). Switch once and read the notice again.
                asked = requested_language(consent_reply)
                other = detected if (detected in ("th", "en") and detected != self.active_language
                                     and conf >= 0.8 and consent_reply) else None
                if not switched and ((asked and asked != self.active_language) or other):
                    switched = True
                    self.active_language = asked or other
                    self.case_container["language"] = self.active_language
                    logger.info(f"[PDPA] switching language to {self.active_language}")
                    if asked or decision == UNCLEAR:
                        replay_notice = True
                        continue
                attempts += 1
                logger.info(f"[PDPA] attempt {attempts}: {decision}")
                if decision != UNCLEAR:
                    break

            if decision == YES:
                self.case_container["conversation"].append({"role": "pdpa_response", "text": consent_reply})
                self.case_container["pdpa_consented"] = True
                self.current_state = KioskState.SYMPTOM_INTERVIEW
            else:
                reason = "consent_refused" if decision == NO else "consent_unclear"
                logger.info(f"No PDPA consent ({reason}). Transitioning to MANUAL_ROUTING...")
                self.refusal_handoff = prepare_pdpa_refusal_handoff(self.case_container, reason=reason)
                with self.telemetry_lock:
                    self.latest_telemetry = {}
                self.current_state = KioskState.MANUAL_ROUTING

        # -------------------------------------------------------------
        # 3b. MANUAL_ROUTING
        # -------------------------------------------------------------
        elif self.current_state == KioskState.MANUAL_ROUTING:
            self.audio.play_template("pdpa_denied", lang=self.active_language, block=True)
            handoff = getattr(self, "refusal_handoff", None)
            if handoff:
                try:
                    self.conveyor_belt.enqueue_nurse_assistance(handoff)
                except OSError as e:
                    logger.error("Dashboard handoff could not be queued. "
                                 f"Follow the audio instruction to contact the nurse directly: {e}")
                self.refusal_handoff = None
            time.sleep(2.0)
            self.cooldown_until = time.time() + SystemConfig.POST_CASE_COOLDOWN_SECONDS
            logger.info("Patient routed to human desk. Resetting to PREPARE_STANDBY.")
            self.current_state = KioskState.PREPARE_STANDBY

        # -------------------------------------------------------------
        # 4. SYMPTOM_INTERVIEW
        # -------------------------------------------------------------
        elif self.current_state == KioskState.SYMPTOM_INTERVIEW:
            self._run_interview()
            if self.case_container.get("exit_reason") == "USER_CANCELLED":
                self.current_state = KioskState.DISPATCH_AND_RESET
            else:
                self.current_state = KioskState.FINALIZING_CASE

        # -------------------------------------------------------------
        # 5. FINALIZING_CASE
        # -------------------------------------------------------------
        elif self.current_state == KioskState.FINALIZING_CASE:
            raw_exit = self.case_container.get("exit_reason") or ExitReason.NORMAL_COMPLETE.value
            exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)
            self.case_container["exit_reason"] = exit_str

            logger.info(f"Packaging case [{self.case_container['case_id']}] (Exit: {exit_str})...")
            try:
                self._update_cumulative_visuals()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not refresh visual signs: {e}")

            # Emergencies skip the slow LLM so the case reaches the dashboard in < 1 s.
            fast = exit_str in (ExitReason.EMERGENCY_INTERRUPT.value, ExitReason.AMS_3_STRIKES.value,
                                ExitReason.SYSTEM_FAULT.value)
            if exit_str == ExitReason.NORMAL_COMPLETE.value and not self.wrapup_played:
                # The patient hears "your case was sent, please wait" while the LLM writes the note.
                self.audio.play_template("wrapup", lang=self.active_language, block=False)
                self.wrapup_played = True
            try:
                verdict = self.reasoning.evaluate_final_case(
                    conversation_history=self.case_container["conversation"],
                    visual_signs=self.case_container["cumulative_visual_signs"],
                    skin_status=self.case_container["facial_skin_status"],
                    exit_reason=exit_str,
                    language=self.active_language,
                    preempt_esi=self.case_container.get("preempt_esi"),
                    use_llm=not fast,
                    intake=self.case_container.get("intake"),
                )
            except Exception as e:  # noqa: BLE001  (never lose the case)
                logger.error(f"Verdict generation failed, using a safe default: {e}")
                verdict = {"esi_level": 2, "urgency": "Emergent [KIOSK ERROR: nurse must assess]",
                           "subjective": " ".join(t["text"] for t in self.case_container["conversation"]
                                                  if t.get("role") == "patient"),
                           "objective": f"Visual signs: {self.case_container['cumulative_visual_signs']}",
                           "assessment": "Automatic assessment failed.", "plan": "Nurse to assess directly.",
                           "audit_mode": f"KIOSK_ERROR_DEFAULT_{type(e).__name__.upper()}"}
            self.case_container["triage_verdict"] = verdict
            logger.info(f"[VERDICT]: ESI Level {verdict.get('esi_level')} ({verdict.get('urgency')})")
            self.current_state = KioskState.DISPATCH_AND_RESET

        # -------------------------------------------------------------
        # 6. DISPATCH_AND_RESET
        # -------------------------------------------------------------
        elif self.current_state == KioskState.DISPATCH_AND_RESET:
            exit_str = str(self.case_container.get("exit_reason") or ExitReason.NORMAL_COMPLETE.value)
            # Queue first: the dashboard alert must not wait for the closing audio.
            self.conveyor_belt.enqueue_case(self.case_container)
            if exit_str == ExitReason.EMERGENCY_INTERRUPT.value:
                self._alert_now("emergency_alert")
            elif exit_str in (ExitReason.AMS_3_STRIKES.value, ExitReason.SYSTEM_FAULT.value,
                              "HUMAN_NURSE_SUMMONED"):
                self._alert_now("ams_alert")
            elif exit_str != "USER_CANCELLED" and not self.wrapup_played:
                self.audio.play_template("wrapup", lang=self.active_language, block=True)
            self._wait_audio_idle()
            # Do not start a new encounter with the same person (e.g. "ขอบคุณค่ะ" after wrap-up).
            self.cooldown_until = time.time() + SystemConfig.POST_CASE_COOLDOWN_SECONDS
            logger.info("Case finalized. Kiosk returning to PREPARE_STANDBY.")
            self.current_state = KioskState.PREPARE_STANDBY

    def _run_interview(self):
        if SystemConfig.INTERVIEW_MODE == "llm":
            self._llm_interview()
        else:
            self._structured_interview()

    # ------------------------------------------------------------------
    # Structured interview (default): complaint -> 0-10 score -> onset -> anything else
    # ------------------------------------------------------------------
    def _say_fixed(self, text: str, role: str = "kiosk_agent") -> None:
        kind = self.audio.speak_fixed(text, lang=self.active_language, block=True)
        spoken = text if kind != "template" else SystemConfig.SCRIPT_TEXTS[self.active_language]["inquiry"]
        self.case_container["conversation"].append({"role": role, "text": spoken})

    def _arm_watchdog(self) -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(SystemConfig.WATCHDOG_SECONDS, repeat=False, file=sys.stderr)

    def _finish_with_anything_else(self, intake: "StructuredIntake | None" = None) -> None:
        """Keep asking "anything else?" until the patient says they are finished.

        Ends on: a finishing phrase ("ไม่มีแล้วค่ะ", "that's all", "no", "อยากพบพยาบาล"),
        two silences in a row, or ANYTHING_ELSE_MAX_ROUNDS extra statements (safety cap).
        Every extra statement is added to the case and checked for emergency words.
        """
        more = SystemConfig.SPOKEN_PROMPTS[self.active_language]["anything_else_more"]
        self._play_logged("anything_else")
        rounds = silences = 0
        while self.running:
            self._arm_watchdog()
            reply = self.audio.listen(language=self.active_language)
            if not reply:
                silences += 1
                if silences >= 2:
                    logger.info("No reply to 'anything else' twice. Finishing.")
                    break
                self._say_fixed(SystemConfig.SPOKEN_PROMPTS[self.active_language]["anything_else_more"])
                continue
            silences = 0
            self.case_container["conversation"].append({"role": "patient", "text": reply})
            if SystemConfig.LOG_TRANSCRIPTS:
                logger.info(f"[Patient]: \"{reply}\"")

            is_preempted, preempt_esi = self.reasoning.check_acute_preemption(
                reply, self.case_container["cumulative_visual_signs"])
            if is_preempted:
                self._emergency_stop(preempt_esi, "statement after 'anything else'")
                return
            with self.telemetry_lock:
                live_gesture = self.latest_telemetry.get("dominant_current_gesture", "NONE")
            if self.reasoning.is_live_chest_emergency(reply, live_gesture):
                self._emergency_stop(2, "chest clutching + severe pain")
                return
            if is_cancel_request(reply):
                self._say_fixed(SystemConfig.SPOKEN_PROMPTS[self.active_language]["cancel_ack"],
                                role="kiosk_action")
                self.case_container["exit_reason"] = "USER_CANCELLED"
                return
            if is_ready_to_finish(reply):
                logger.info("Patient said they are finished.")
                break

            if intake is not None:                      # may fill a missing score / onset
                intake.update(reply)
                self.case_container["intake"] = intake.as_dict()
            rounds += 1
            if rounds >= SystemConfig.ANYTHING_ELSE_MAX_ROUNDS:
                logger.info("Reached the maximum number of extra statements. Finishing.")
                break
            self._say_fixed(more)
        self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value

    def _structured_interview(self):
        intake = StructuredIntake(max_asks=SystemConfig.INTAKE_MAX_ASKS)
        with self.telemetry_lock:
            dom_gesture = self.latest_telemetry.get("dominant_current_gesture", "NONE")
        proactive_prompt = self.reasoning.get_proactive_visual_prompt(dom_gesture, lang=self.active_language)
        if proactive_prompt:
            logger.info(f"[Proactive Visual Cue]: Triggering inquiry for {dom_gesture}")
            self._say_fixed(proactive_prompt, role="kiosk_proactive")
        else:
            self._play_logged("inquiry")
        intake.mark_asked("complaint")

        pauses = 0
        try:
            while self.running:
                self._arm_watchdog()
                turn_start = time.time()
                chunk = self.audio.listen(language=self.active_language)
                stt_duration = time.time() - turn_start
                self._update_cumulative_visuals()

                if not chunk:
                    pauses += 1
                    if pauses == 1:
                        self._play_logged("still_listening")
                        continue
                    if pauses == 2 and intake.last_asked:
                        nxt = intake.next_question(self.active_language)   # clearer re-ask
                        if nxt:
                            self._say_fixed(nxt[1])
                            continue
                    if pauses == 2:
                        self._play_logged("anything_else")
                        continue
                    logger.info("Extended silence verified. Finalizing encounter.")
                    self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                    return
                pauses = 0
                self.case_container["conversation"].append({"role": "patient", "text": chunk})
                if SystemConfig.LOG_TRANSCRIPTS:
                    logger.info(f"[Patient]: \"{chunk}\"")

                is_preempted, preempt_esi = self.reasoning.check_acute_preemption(
                    chunk, self.case_container["cumulative_visual_signs"])
                if is_preempted:
                    self._emergency_stop(preempt_esi, "patient statement / chest posture")
                    return
                with self.telemetry_lock:
                    live_gesture = self.latest_telemetry.get("dominant_current_gesture", "NONE")
                if self.reasoning.is_live_chest_emergency(chunk, live_gesture):
                    self._emergency_stop(2, "chest clutching + severe pain")
                    return
                if is_cancel_request(chunk):
                    logger.info("Patient requested cancellation.")
                    self._say_fixed(SystemConfig.SPOKEN_PROMPTS[self.active_language]["cancel_ack"],
                                    role="kiosk_action")
                    self.case_container["exit_reason"] = "USER_CANCELLED"
                    return
                if self.reasoning.is_stopping_phrase(chunk, lang=self.active_language):
                    logger.info("Patient explicitly requested completion.")
                    self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                    return

                new_lang = requested_language(chunk)
                if new_lang and new_lang != self.active_language:
                    logger.info(f"Switching language {self.active_language} -> {new_lang}")
                    self.active_language = new_lang
                    self.case_container["language"] = new_lang
                    slot = intake.last_asked or "complaint"
                    intake.asks[slot] = max(0, intake.asks[slot] - 1)   # ask it again, in the new language
                    nxt = intake.next_question(new_lang)
                    if nxt:
                        self._say_fixed(nxt[1])
                    continue

                answered = intake.update(chunk)
                self.case_container["intake"] = intake.as_dict()
                logger.info(f"[Intake]: answered={answered} {intake.as_dict()}")

                if intake.unusable_in_a_row >= SystemConfig.MAX_OFFTOPIC_STRIKES:
                    logger.warning("3 answers in a row did not answer the question: possible altered mental status.")
                    self.case_container["exit_reason"] = ExitReason.AMS_3_STRIKES.value
                    return

                nxt = intake.next_question(self.active_language)
                self.case_container["intake"] = intake.as_dict()
                t_tts = time.time()
                if nxt is None:
                    logger.info("Intake questions finished. Asking 'anything else'.")
                    self._finish_with_anything_else(intake)
                    return
                self._say_fixed(nxt[1])
                logger.info(f"[TELEMETRY]: Listen+STT: {stt_duration:.2f}s | LLM: 0.00s (structured) | "
                            f"TTS: {time.time() - t_tts:.2f}s | Turnaround: {time.time() - turn_start:.2f}s")
        finally:
            faulthandler.cancel_dump_traceback_later()

    # ------------------------------------------------------------------
    # Original LLM-agent interview (INTERVIEW_MODE=llm)
    # ------------------------------------------------------------------
    def _llm_interview(self):
        with self.telemetry_lock:
            dom_gesture = self.latest_telemetry.get("dominant_current_gesture", "NONE")

        proactive_prompt = self.reasoning.get_proactive_visual_prompt(dom_gesture, lang=self.active_language)
        if proactive_prompt:
            logger.info(f"[Proactive Visual Cue]: Triggering inquiry for {dom_gesture}")
            self._say(proactive_prompt, "inquiry", role="kiosk_proactive")
        else:
            self._play_logged("inquiry")

        consecutive_pauses = 0
        try:
            self._interview_loop(consecutive_pauses)
        finally:
            faulthandler.cancel_dump_traceback_later()

    def _interview_loop(self, consecutive_pauses: int):
        while self.running:
            # Watchdog: if this turn freezes, print where every thread is stuck.
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(SystemConfig.WATCHDOG_SECONDS, repeat=False, file=sys.stderr)
            turn_start_time = time.time()
            t_stt_start = time.time()
            chunk = self.audio.listen(language=self.active_language)
            stt_duration = time.time() - t_stt_start
            self._update_cumulative_visuals()

            # Silence handling
            if len(chunk) == 0:
                consecutive_pauses += 1
                if consecutive_pauses == 1:
                    self._play_logged("still_listening")
                    continue
                if consecutive_pauses == 2:
                    self._play_logged("anything_else")
                    continue
                logger.info("Extended silence verified. Finalizing encounter.")
                self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                return

            consecutive_pauses = 0
            self.case_container["conversation"].append({"role": "patient", "text": chunk})

            # Emergency preemption: alert NOW, before anything slow
            is_preempted, preempt_esi = self.reasoning.check_acute_preemption(
                chunk, self.case_container["cumulative_visual_signs"])
            if is_preempted:
                self._emergency_stop(preempt_esi, "patient statement / chest posture")
                return

            # "That's all" (whole reply only)
            if self.reasoning.is_stopping_phrase(chunk, lang=self.active_language):
                logger.info("Patient explicitly requested completion.")
                self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                return

            with self.telemetry_lock:
                v_telemetry = self.latest_telemetry

            action_frame = self.reasoning.dispatch_agent_turn(
                utterance=chunk,
                history=self.case_container["conversation"][:-1],
                visual_telemetry=v_telemetry,
                current_strikes=self.current_strikes,
                active_lang=self.active_language,
            )
            action = action_frame.get("action", "CONTINUE_TRIAGE")
            spoken_resp = action_frame.get("spoken_response", "")
            llm_duration = action_frame.get("llm_latency_sec", 0.0)
            self.current_strikes = action_frame.get("strikes", self.current_strikes)
            logger.info(f"[Agent Decision]: Action={action} | Spoken=\"{spoken_resp}\"")

            t_tts_start = time.time()
            if action == "TRIAGE_COMPLETE":
                logger.info("All necessary clinical data gathered. Concluding interview.")
                self._finish_with_anything_else()
                return

            if action == "CANCEL_AND_EXIT":
                logger.info("Patient requested early cancellation.")
                self._say(spoken_resp, "wrapup", role="kiosk_action")
                self.case_container["exit_reason"] = "USER_CANCELLED"
                return

            if action == "CALL_HUMAN_NURSE":
                logger.warning("Summoning human triage staff.")
                self.case_container["exit_reason"] = "HUMAN_NURSE_SUMMONED"
                return

            if action == "EMERGENCY_ABORT":
                self._emergency_stop(2, "agent rule")
                return

            if action == "TRIGGER_AMS":
                logger.warning("3-Strike Rule Tripped: Altered Mental Status (AMS).")
                self.case_container["exit_reason"] = ExitReason.AMS_3_STRIKES.value
                return

            if action == "SWITCH_LANGUAGE":
                new_lang = action_frame.get("target_language", "en")
                logger.info(f"Switching language {self.active_language} -> {new_lang}")
                self.active_language = new_lang
                self.case_container["language"] = new_lang
                self._say(spoken_resp, "inquiry", role="kiosk_action")
            elif action_frame.get("is_off_topic", False):
                self.audio.speak_off_topic(lang=self.active_language, block=True)
                self.case_container["conversation"].append(
                    {"role": "kiosk_agent", "text": safe_off_topic_text(self.active_language)})
            else:
                self._say(spoken_resp, "inquiry")

            tts_duration = time.time() - t_tts_start
            total_turnaround = time.time() - turn_start_time
            logger.info(f"[TELEMETRY]: Listen+STT: {stt_duration:.2f}s | LLM: {llm_duration:.2f}s | "
                        f"TTS: {tts_duration:.2f}s | Turnaround: {total_turnaround:.2f}s")

    def run(self):
        """Main-Thread GUI Capture Loop (macOS Cocoa Compliant)."""
        cap = open_camera(SystemConfig.CAMERA_INDEX)
        if cap is None:
            self.running = False
            self.conveyor_belt.running = False
            self.vision.release()
            self.audio.release()
            raise RuntimeError(
                f"Could not open camera device {SystemConfig.CAMERA_INDEX}. Close other apps that use the camera, "
                "allow camera access for this terminal in the OS privacy settings, or try CAMERA_INDEX=1."
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
                    # Clean camera window hidden for now (kept for later use):
                    # cv2.imshow("Kiosk - Clean Patient View", telemetry["clean_view"])
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
    configure_console_utf8()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    kiosk = SmartReceptionTriageKiosk()
    kiosk.run()