import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from triage_kiosk.triage.reasoning import ReasoningEngine

GOOD = {"esi_level": 4, "urgency": "Less Urgent", "subjective": "s", "objective": "o",
        "assessment": "a", "plan": "p"}


def ollama(body, status=200):
    return SimpleNamespace(status_code=status, json=lambda: {"response": json.dumps(body)})


class FinalVerdictSafetyTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine.__new__(ReasoningEngine)
        self.engine.endpoint = "http://127.0.0.1:11434/api/generate"

    def run_case(self, exit_reason="NORMAL_COMPLETE", visual=None, skin="NORMAL", text="mild headache", **kw):
        return self.engine.evaluate_final_case([{"role": "patient", "text": text}], visual or {}, skin,
                                               exit_reason, "en", **kw)

    @patch("triage_kiosk.triage.reasoning.requests.post", side_effect=requests.Timeout("t"))
    def test_emergency_with_llm_timeout_is_esi_1_or_2(self, _p):
        self.assertEqual(self.run_case("EMERGENCY_INTERRUPT", text="แน่นหน้าอกมาก")["esi_level"], 2)
        self.assertEqual(self.run_case("EMERGENCY_INTERRUPT", preempt_esi=1)["esi_level"], 1)

    @patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama(GOOD))
    def test_llm_cannot_under_triage_chest_clutching(self, _p):
        v = self.run_case(visual={"chest_clutching": True})
        self.assertEqual(v["esi_level"], 2)
        self.assertEqual(v["esi_before_safety_floor"], 4)

    @patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama(GOOD))
    def test_llm_verdict_kept_when_no_hard_signal(self, _p):
        v = self.run_case()
        self.assertEqual((v["esi_level"], v["audit_mode"]), (4, "LLM_INFERENCE_VALIDATED"))

    @patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama(GOOD))
    def test_flushing_caps_at_3_and_ams_at_2(self, _p):
        self.assertEqual(self.run_case(skin="RED_FLUSHING")["esi_level"], 3)
        self.assertEqual(self.run_case("AMS_3_STRIKES")["esi_level"], 2)

    @patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama(GOOD, status=404))
    def test_http_error_uses_fallback_instead_of_crashing(self, _p):
        v = self.run_case()
        self.assertIn("OLLAMA_HTTP_404", v["audit_mode"])

    def test_invalid_llm_verdicts_are_rejected(self):
        for bad in [{"esi_level": 0}, {"esi_level": 6}, {"esi_level": "two"}, {"urgency": "x"}, [1, 2],
                    {"esi_level": 2.5}, {"esi_level": True}]:
            with patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama(bad)):
                v = self.run_case()
            self.assertIn(v["esi_level"], (1, 2, 3, 4, 5), bad)
            self.assertIn("INVALID_LLM_VERDICT", v["audit_mode"], bad)

    @patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama({"esi_level": "3"}))
    def test_missing_soap_fields_are_filled(self, _p):
        v = self.run_case()
        self.assertEqual(v["esi_level"], 3)
        for k in ("urgency", "subjective", "objective", "assessment", "plan"):
            self.assertTrue(v[k])

    @patch("triage_kiosk.triage.reasoning.requests.post")
    def test_emergency_fast_path_skips_llm(self, post):
        v = self.run_case("EMERGENCY_INTERRUPT", use_llm=False)
        post.assert_not_called()
        self.assertEqual(v["esi_level"], 2)

    @patch("triage_kiosk.triage.reasoning.requests.post", side_effect=requests.ConnectionError("x"))
    def test_cold_sweat_is_not_a_minor_cold(self, _p):
        self.assertEqual(self.run_case(text="I have a cold sweat")["esi_level"], 4)
        self.assertEqual(self.run_case(text="I have a cold and runny nose")["esi_level"], 5)


class TurnGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine.__new__(ReasoningEngine)
        self.engine.endpoint = "http://x"

    def turn(self, text, body):
        with patch("triage_kiosk.triage.reasoning.requests.post", return_value=ollama(body)):
            return self.engine.dispatch_agent_turn(text, [], {}, 0, "en")

    def test_patient_cancel_words(self):
        self.assertEqual(self.turn("cancel please", {})["action"], "CANCEL_AND_EXIT")
        self.assertNotEqual(self.turn("I can't stop coughing",
                                      {"action": "CONTINUE_TRIAGE", "spoken_response": "Since when?"})["action"],
                            "CANCEL_AND_EXIT")

    def test_llm_cannot_cancel_or_invent_actions(self):
        for action in ["CANCEL_AND_EXIT", "DELETE_PATIENT", None]:
            f = self.turn("my stomach hurts", {"action": action, "spoken_response": "Since when?"})
            self.assertEqual(f["action"], "CONTINUE_TRIAGE", action)

    def test_empty_spoken_response_uses_fallback_question(self):
        f = self.turn("my stomach hurts", {"action": "CONTINUE_TRIAGE", "spoken_response": ""})
        self.assertIn("scale of 1 to 10", f["spoken_response"])


if __name__ == "__main__":
    unittest.main()
