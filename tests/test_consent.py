import unittest

from triage_kiosk.consent import prepare_pdpa_refusal_handoff


class PreparePdpaRefusalHandoffTests(unittest.TestCase):
    def test_clears_unconsented_case_and_returns_minimal_event(self):
        case_container = {
            "case_id": "CASE-123",
            "language": "th",
            "conversation": [{"role": "pdpa_response", "text": "No"}],
            "visual_evidence_path": "/private/evidence.jpg",
            "triage_verdict": {"esi_level": 2},
        }

        event = prepare_pdpa_refusal_handoff(case_container)

        self.assertEqual(case_container, {})
        self.assertEqual(
            event,
            {
                "event_type": "nurse_assistance_requested",
                "reason": "consent_refused",
            },
        )
        self.assertNotIn("case_id", event)
        self.assertNotIn("conversation", event)
        self.assertNotIn("visual_evidence_path", event)


if __name__ == "__main__":
    unittest.main()
