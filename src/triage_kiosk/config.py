"""Central configuration: kiosk states, exit reasons, and system parameters."""

import os
from pathlib import Path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from enum import Enum

class KioskState(Enum):
    PREPARE_STANDBY = "PREPARE_STANDBY"       # Idle monitoring: Camera + Wake-word listener
    GREETING_LANG_SELECT = "GREETING"         # Bilingual introduction & language negotiation
    PDPA_CONSENT = "PDPA_CONSENT"             # Read PDPA policy and await verbal consent
    SYMPTOM_INTERVIEW = "INTERVIEW"           # Interactive OPQRST Probing & Active Nudging Loop
    FINALIZING_CASE = "FINALIZING"           # Holistic multi-modal synthesis & SOAP generation
    DISPATCH_AND_RESET = "DISPATCH"           # Enqueue to conveyor belt & reset kiosk
    MANUAL_ROUTING = "MANUAL_ROUTING"         # Direct routing to human nurse desk (e.g. PDPA denied)


class ExitReason(Enum):
    NORMAL_COMPLETE = "NORMAL_COMPLETE"               # Encounter concluded naturally
    EMERGENCY_INTERRUPT = "EMERGENCY_INTERRUPT"       # Acute distress detected mid-case (ESI 1-2)
    AMS_3_STRIKES = "AMS_3_STRIKES"                   # 3 consecutive off-topic strikes (ESI 2 AMS)
    PDPA_DENIED = "PDPA_DENIED"                       # Patient refused data collection
    ABANDONED_TIMEOUT = "ABANDONED_TIMEOUT"           # Patient walked away mid-encounter
    SYSTEM_FAULT = "SYSTEM_FAULT"                     # Mic/speech/software fault mid-encounter


class SystemConfig:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # --- Video & Display Pipeline ---
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    SHOW_DUAL_WINDOWS = True                 # Clean Patient View + Diagnostic Skeleton View

    # --- Perception Thresholds (Scale-Invariant Torso Ratios) ---
    PRESENCE_FRAMES_TRIGGER = 15             # Frames to confirm patient presence (~0.5s)
    PRESENCE_DETECTION_CONFIRM_FRAMES = 15   # Alias for backward compatibility
    GESTURE_HOLD_SECONDS = 1.5               # Scratch Filter: Hold clutching pose > 1.5s
    SKIN_DUTY_CYCLE_SECONDS = 7.0            # Run CNN skin inference once every 7.0 seconds
    GESTURE_SUSTAINED_SECONDS = 3.0          # Must sustain gesture for >=3.0s to count in final case
    
    # Anatomical distance ratios relative to torso length (|Mid-Shoulder - Mid-Hip|)
    CHEST_PAIN_TORSO_RATIO = 0.35            # Wrist to Sternum relative to Torso length
    ABDOMEN_PAIN_TORSO_RATIO = 0.40          # Wrist to Navel relative to Torso length
    HEADACHE_HEAD_RATIO = 0.28               # Wrist to Temple/Nose relative to Torso length
    ELBOW_FLEXION_MAX_ANGLE = 115.0          # Degrees: Arm must be bent inward
    CROSSED_ARMS_WRIST_RATIO = 0.45          # Crossed Arms Filter ratio
    THINKING_FOREARM_MIN_ANGLE = 65.0        # Thinking Pose vertical forearm angle

    # Posture rules: "v2" = corrected rules (default), "v1" = original rules (kept for comparison).
    POSTURE_RULES_VERSION = os.getenv("POSTURE_RULES_VERSION", "v2")
    # Optional overrides of the v2 values in vision/posture.py (PostureConfig); tuned on dev images only.
    POSTURE_V2_PARAMS: dict = {}

    # --- TensorFlow Lite / CNN Skin Model ---
    TFLITE_SKIN_MODEL_PATH = str(PROJECT_ROOT / "models" / "skin_classifier.tflite")
    SKIN_CONFIDENCE_THRESHOLD = 0.65         # Used ONLY for a model without metadata (the original model).
                                             # Converted models carry their validation threshold in models/skin_classifier.json
    # "patch" = forehead/cheek patches as in CNN training (default); "legacy_whole_face" = original behaviour.
    SKIN_PIPELINE = os.getenv("SKIN_PIPELINE", "patch")
    # 1 = stop with an error if the skin model cannot be loaded (use in experiments).
    SKIN_REQUIRE_MODEL = os.getenv("SKIN_REQUIRE_MODEL", "0").strip().lower() in {"1", "true", "yes"}

    # --- Audio Subsystem: Wake-Word & Fast-Whisper ---
    WAKE_WORDS_TH = ["สวัสดี", "หวัดดี", "เริ่ม", "ช่วยด้วย", "ตรวจ", "ฮัลโหล"]
    WAKE_WORDS_EN = ["hello", "hi", "start", "help", "triage", "greeting", "hey"]

# Set physical mic to Index 1 (Your USB Mic), but i use 0 for defult mac mic
    # None = the system's default input device (set MIC_DEVICE_INDEX to pick another one)
    MICROPHONE_DEVICE_INDEX = int(os.environ["MIC_DEVICE_INDEX"]) if os.getenv("MIC_DEVICE_INDEX") else None

# Force CPU for Fast-Whisper (Zero CUDA DLL errors & frees VRAM for Llama) 
    FASTER_WHISPER_DEVICE = "cpu"
    FASTER_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")           # Local Fast-Whisper model
    FASTER_WHISPER_COMPUTE = "int8"          # 8-bit quantization: lightweight (~600MB memory)
    FASTER_WHISPER_CPU_THREADS = 4

    SPEECH_PAUSE_THRESHOLD = float(os.getenv("SPEECH_PAUSE_THRESHOLD", "1.2"))  # silence that ends a reply
    SPEECH_TIMEOUT = 10.0                    # Max wait time for patient speech initiation
    # Max length of one reply. In the first Mac test most replies ran to the old 20 s limit
    # because room noise kept the energy detector "on"; 12 s is enough for one answer.
    SPEECH_PHRASE_LIMIT = float(os.getenv("SPEECH_PHRASE_LIMIT", "12"))
    MIC_ENERGY_THRESHOLD = int(os.getenv("MIC_ENERGY_THRESHOLD", "300"))   # starting sensitivity floor
    MIC_DYNAMIC_ENERGY = os.getenv("MIC_DYNAMIC_ENERGY", "1").strip() in {"1", "true", "yes"}
    MIC_DYNAMIC_RATIO = float(os.getenv("MIC_DYNAMIC_RATIO", "2.0"))      # speech must be this x louder than noise
    MIC_CALIBRATE_SECONDS = 0.5              # re-measure room noise at the start of every encounter

    # Whisper decoding limits (keep decode time bounded; see audio/engine.py)
    WHISPER_MAX_NEW_TOKENS = int(os.getenv("WHISPER_MAX_NEW_TOKENS", "160"))

    # A single interview turn (listen + STT + LLM + speech) that takes longer than this dumps
    # every thread's stack trace to the terminal, so a freeze shows exactly where it is stuck.
    WATCHDOG_SECONDS = float(os.getenv("WATCHDOG_SECONDS", "60"))

    # 1 = print what Whisper heard at INFO level (for testing only; patient words appear in the terminal)
    LOG_TRANSCRIPTS = os.getenv("KIOSK_LOG_TRANSCRIPTS", "0").strip() in {"1", "true", "yes"}

    # --- Conversational Clinical Rules & OPQRST ---
    MAX_OFFTOPIC_STRIKES = 3                 # 3 strikes = Altered Mental Status (ESI Level 2)
    PROBE_MAX_TOKENS = 35                    # Strictly cap follow-up question generation latency
    
    # Stopping Intent Tokens
    STOP_PHRASES_TH = ["หมดแล้ว", "แค่นี้", "ไม่มีแล้ว", "พอแล้ว", "เสร็จแล้ว", "ไม่มีอะไรเพิ่ม", "มีเท่านี้"]
    STOP_PHRASES_EN = ["that's all", "thats all", "nothing else", "no more", "done", "finished", "i'm good"]

    # --- Audio Template Cache Directory ---
    BASE_DIR = str(PROJECT_ROOT)
    AUDIO_TEMPLATE_DIR = str(PROJECT_ROOT / "assets" / "audio")
    # Pre-rendered audio of every fixed sentence (scripts/audio/generate_templates.py)
    FIXED_AUDIO_DIR = str(PROJECT_ROOT / "assets" / "audio" / "fixed")

    # Audio Template File Mapping
    TEMPLATES = {
        "th": {
            "greeting": "th_greeting.mp3",
            "pdpa_notice": "th_pdpa.mp3",
            "pdpa_denied": "th_pdpa_denied.mp3",
            "inquiry": "th_inquiry.mp3",
            "still_listening": "th_still_listening.mp3",
            "anything_else": "th_anything_else.mp3",
            "wrapup": "th_wrapup.mp3",
            "emergency_alert": "th_emergency_alert.mp3",
            "ams_alert": "th_ams_alert.mp3"
        },
        "en": {
            "greeting": "en_greeting.mp3",
            "pdpa_notice": "en_pdpa.mp3",
            "pdpa_denied": "en_pdpa_denied.mp3",
            "inquiry": "en_inquiry.mp3",
            "still_listening": "en_still_listening.mp3",
            "anything_else": "en_anything_else.mp3",
            "wrapup": "en_wrapup.mp3",
            "emergency_alert": "en_emergency_alert.mp3",
            "ams_alert": "en_ams_alert.mp3"
        }
    }

    # Text Sources for Audio Templates and Picture Path (Polite Clinical Tone)
    SCRIPT_TEXTS = {
        "th": {
            "greeting": "สวัสดีค่ะ ดิฉันคือระบบผู้ช่วยพยาบาลคัดกรองอัตโนมัติ กรุณาแจ้งอาการหรือเลือกใช้งานเป็นภาษาไทยหรือภาษาอังกฤษได้เลยค่ะ",
            "pdpa_notice": "เนื่องด้วยตามพ.ร.บ.คุ้มครองข้อมูลส่วนบุคคลหรือPDPAระบบขออนุญาตบันทึกเสียงและภาพของการสนทนานี้ เพื่อการคัดกรองทางการแพทย์ ท่านประสงค์ยินยอมหรือไม่คะ?",
            "pdpa_denied": "รับทราบค่ะ ระบบจะไม่บันทึกข้อมูลของท่านใดๆ กรุณาติดต่อเคาน์เตอร์พยาบาลด้านหน้าได้โดยตรง ขอบคุณค่ะ",
            "inquiry": "วันนี้มีอาการไม่สบายตรงไหนหรือมีอะไรให้ดิฉันช่วยดูแลคะ?",
            "still_listening": "ดิฉันยังฟังอยู่นะคะ เล่าต่อได้เลยค่ะ",
            "anything_else": "ระบบบันทึกข้อมูลเรียบร้อยค่ะ มีอะไรเพิ่มเติมอีกไหมคะ?",
            "wrapup": "ระบบได้รวบรวมข้อมูลและอาการทั้งหมดและส่งไปยังพยาบาลเรียบร้อยแล้ว กรุณานั่งรอเรียกชื่อที่จุดพักคอยนะคะ",
            "emergency_alert": "ระบบตรวจพบอาการฉุกเฉิน กำลังส่งสัญญาณเรียกพยาบาลเข้ามาดูแลทันที กรุณานั่งนิ่งๆก่อนนะคะ",
            "ams_alert": "ระบบได้ส่งสัญญาณแจ้งเตือนพยาบาลให้เข้ามาดูแลท่านที่หน้าตู้คัดกรองแล้ว กรุณารอสักครู่ค่ะ"
        },
        "en": {
            "greeting": "Hello, I am your automated triage nurse assistant system. Please select English or Thai to proceed.",
            "pdpa_notice": "Under privacy regulations, the system requests permission to record audio and video of this conversation for medical screening purposes. Do you consent?",
            "pdpa_denied": "Understood. No data will be stored. Please proceed directly to the nurse reception desk. Thank you.",
            "inquiry": "What symptoms or discomfort bring you to the emergency department today? What can I help you with?",
            "still_listening": "I am still listening. Please feel free to continue.",
            "anything_else": "I have noted that down. Do you have any else or details to add?",
            "wrapup": "Your clinical case has been compiled and dispatched to the triage nurse. Please take a seat in the waiting area.",
            "emergency_alert": "Critical emergency detected. Alerting resuscitation nurses to your station immediately. Please remain seated.",
            "ams_alert": "Triage staff have been dispatched to assist you at the kiosk. Please wait a moment."
        }
    }

    # Short prompts spoken with the offline voice (no pre-recorded file exists for these).
    SPOKEN_PROMPTS = {
        "th": {
            "pdpa_reask": "ขออภัยค่ะ ดิฉันไม่แน่ใจคำตอบ กรุณาตอบว่า ยินยอม หรือ ไม่ยินยอม ค่ะ",
            "anything_else_more": "มีอาการหรือเรื่องอื่นที่อยากบอกเพิ่มเติมอีกไหมคะ ถ้าไม่มีแล้ว ตอบว่า ไม่มีแล้ว ได้เลยค่ะ",
            "cancel_ack": "รับทราบค่ะ ยกเลิกการคัดกรองแล้ว หากต้องการความช่วยเหลือ กรุณาติดต่อเคาน์เตอร์พยาบาลได้เลยค่ะ",
        },
        "en": {
            "pdpa_reask": "Sorry, I did not catch that. Please say: yes, I agree. Or: no, I do not agree.",
            "anything_else_more": "Is there anything else you would like to add? If not, just say that's all.",
            "cancel_ack": "Understood. The screening is cancelled. Please go to the nurse desk if you need help.",
        },
    }

    TIME_AND_NUMERIC_KEYWORDS = [
        "โมง", "เช้า", "สาย", "บ่าย", "เย็น", "ค่ำ", "ดึก", "วัน", "ชั่วโมง", "นาที", 
        "เมื่อวาน", "ก่อน", "หลัง", "ตี", "เมื่อกี้", "เมื่อคืน", "ตั้งแต่",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "morning", "afternoon", "evening", "night", "yesterday", "hours", "days", "mins", "since",
        "scale", "mild", "severe", "moderate"
    ]

    EVIDENCE_DIR = str(PROJECT_ROOT / "data" / "runtime" / "evidence")

    # --- Reasoning Engine (Local Ollama Llama 3.2) ---
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = "llama3.2:1b" if os.getenv("USE_1B") else "llama3.2:latest"
    # Model for the short follow-up questions only (the final ESI always uses OLLAMA_MODEL).
    # e.g. OLLAMA_TURN_MODEL=llama3.2:1b roughly halves the per-question wait.
    OLLAMA_TURN_MODEL = os.getenv("OLLAMA_TURN_MODEL", OLLAMA_MODEL)
    ALLOW_CLOUD_SPEECH_RECOGNITION = (
        os.getenv("ALLOW_CLOUD_SPEECH_RECOGNITION", "0").strip().lower()
        in {"1", "true", "yes"}
    )
    # Timeouts before the deterministic fallback is used. Round A measured a median of
    # ~7.3 s for the final verdict on the M1, so the old single 6 s timeout made most
    # real calls fall back. Override with env vars if the kiosk machine is faster/slower.
    OLLAMA_TURN_TIMEOUT = float(os.getenv("OLLAMA_TURN_TIMEOUT", "10"))    # one interview question
    # Final ESI + SOAP note. The patient hears the wrap-up message first, so this wait is hidden.
    OLLAMA_FINAL_TIMEOUT = float(os.getenv("OLLAMA_FINAL_TIMEOUT", "45"))
    OLLAMA_TIMEOUT = OLLAMA_TURN_TIMEOUT                                   # kept for older scripts

    # --- Offline text-to-speech for the LLM's follow-up questions ---
    # macOS `say` runs on the device (no audio or text leaves the kiosk). The Thai voice
    # "Kanya" may need to be downloaded once: System Settings > Accessibility >
    # Spoken Content > System voice > Manage Voices > Thai.
    LOCAL_TTS_ENABLED = os.getenv("LOCAL_TTS", "1").strip().lower() in {"1", "true", "yes"}
    LOCAL_TTS_VOICES = {
        "th": os.getenv("LOCAL_TTS_VOICE_TH", "Kanya"),
        "en": os.getenv("LOCAL_TTS_VOICE_EN", "Samantha"),
    }
    LOCAL_TTS_RATE = int(os.getenv("LOCAL_TTS_RATE", "175"))   # words per minute
    # Windows (SAPI): Thai "Microsoft Pattara" needs the Thai speech pack:
    # Settings > Time & language > Language & region > Thai > Language options > Speech.
    LOCAL_TTS_VOICES_WINDOWS = {
        "th": os.getenv("LOCAL_TTS_VOICE_TH_WIN", "Pattara"),
        "en": os.getenv("LOCAL_TTS_VOICE_EN_WIN", "Zira"),
    }

    # --- Fault handling ---
    FSM_MAX_CONSECUTIVE_ERRORS = 3            # then: tell staff (dashboard) and reset to standby
    FSM_FAULT_BACKOFF_SECONDS = 10.0
    POST_CASE_COOLDOWN_SECONDS = 15.0         # same person still in front after wrap-up: don't restart
    # "structured" = fixed step-by-step questions (complaint -> 0-10 score -> onset), default.
    # "llm" = the original free-form LLM agent questions (kept for comparison).
    INTERVIEW_MODE = os.getenv("INTERVIEW_MODE", "structured").strip().lower()
    ANYTHING_ELSE_MAX_ROUNDS = 5              # "anything else?" repeats until "that's all" (cap: 5 extra statements)
    INTAKE_MAX_ASKS = 2                       # each question: asked once + one clearer re-ask, then move on
    PDPA_MAX_ASKS = 2                         # unclear/silent consent -> ask again once, then nurse

    # --- Background "Conveyor Belt" Outbox Spooler ---
    OUTBOX_SPOOL_DIR = str(PROJECT_ROOT / "data" / "runtime" / "spool_outbox")
    NURSE_DASHBOARD_ENDPOINT = os.getenv(
        "NURSE_DASHBOARD_ENDPOINT", "http://127.0.0.1:8000/api/triage/submit_case"
    )
    # Shared secret sent as the X-Kiosk-Key header; must match DASHBOARD_KIOSK_KEY on the dashboard.
    NURSE_DASHBOARD_KEY = os.getenv("NURSE_DASHBOARD_KEY", "")
    HTTP_TIMEOUT = 3.0
    CONVEYOR_RETRY_INTERVAL = 30.0            # Seconds between dispatch retry attempts