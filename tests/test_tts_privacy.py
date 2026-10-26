import unittest

from triage_kiosk.tts_privacy import SAFE_OFF_TOPIC_TEXT, safe_off_topic_text


class SafeOffTopicTextTests(unittest.TestCase):
    def test_only_fixed_generic_text_is_available_for_external_tts(self):
        self.assertEqual(safe_off_topic_text("en"), SAFE_OFF_TOPIC_TEXT["en"])
        self.assertEqual(safe_off_topic_text("th"), SAFE_OFF_TOPIC_TEXT["th"])
        self.assertNotIn("patient", safe_off_topic_text("en").lower())
        self.assertNotIn("CASE-", safe_off_topic_text("en"))

    def test_unsupported_language_fails_closed(self):
        with self.assertRaises(ValueError):
            safe_off_topic_text("fr")


if __name__ == "__main__":
    unittest.main()
