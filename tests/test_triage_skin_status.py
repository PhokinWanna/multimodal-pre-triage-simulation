import unittest
from unittest.mock import patch

import requests

from triage_kiosk.triage.reasoning import ReasoningEngine


class DeterministicSkinStatusTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine.__new__(ReasoningEngine)
        self.engine.endpoint = "http://127.0.0.1:11434/api/generate"

    @patch(
        "triage_kiosk.triage.reasoning.requests.post",
        side_effect=requests.ConnectionError("test: Ollama unavailable"),
    )
    def test_uppercase_normal_model_status_allows_low_risk_fallback(self, _post):
        result = self.engine.evaluate_final_case(
            conversation_history=[{"role": "patient", "text": "mild headache"}],
            visual_signs={},
            skin_status="NORMAL",
            exit_reason="NORMAL_COMPLETE",
            language="en",
        )

        self.assertEqual(result["esi_level"], 4)
        self.assertIn("DETERMINISTIC_FALLBACK", result["audit_mode"])

    @patch(
        "triage_kiosk.triage.reasoning.requests.post",
        side_effect=requests.ConnectionError("test: Ollama unavailable"),
    )
    def test_flushing_model_status_is_not_treated_as_normal(self, _post):
        result = self.engine.evaluate_final_case(
            conversation_history=[{"role": "patient", "text": "mild headache"}],
            visual_signs={},
            skin_status="RED_FLUSHING",
            exit_reason="NORMAL_COMPLETE",
            language="en",
        )

        self.assertNotEqual(result["esi_level"], 4)


if __name__ == "__main__":
    unittest.main()
