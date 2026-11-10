import unittest

from triage_kiosk.triage.intake import QUESTIONS, REPROMPTS, StructuredIntake, extract_onset, extract_severity


class ExtractTests(unittest.TestCase):
    def test_severity_explicit_anywhere(self):
        self.assertEqual(extract_severity("ปวดหัว 8 เต็ม 10")[0], 8)
        self.assertEqual(extract_severity("about 6 out of 10")[0], 6)
        self.assertEqual(extract_severity("7/10")[0], 7)

    def test_bare_number_only_when_asked(self):
        self.assertEqual(extract_severity("7", asked=True)[0], 7)
        self.assertEqual(extract_severity("เจ็ดค่ะ", asked=True)[0], 7)
        self.assertEqual(extract_severity("7", asked=False), (None, None))

    def test_time_is_not_a_pain_score(self):
        self.assertEqual(extract_severity("2 hours ago", asked=True), (None, None))
        self.assertEqual(extract_severity("เป็นมา 3 วัน", asked=True), (None, None))

    def test_severity_words(self):
        self.assertEqual(extract_severity("ปวดหัวมากค่ะ", asked=True), (None, "severe"))
        self.assertEqual(extract_severity("ไม่ค่อยเจ็บ", asked=True), (None, "mild"))
        self.assertEqual(extract_severity("ไม่เจ็บค่ะ แค่มีไข้", asked=True), (None, "none"))

    def test_onset(self):
        self.assertEqual(extract_onset("it started 2 hours ago"), "2 hours")
        self.assertEqual(extract_onset("เป็นตั้งแต่เมื่อวานค่ะ"), "เมื่อวาน")
        self.assertEqual(extract_onset("สามวันแล้ว"), "สามวัน")
        self.assertEqual(extract_onset("ไม่แน่ใจค่ะ", asked=True), "unknown")
        self.assertIsNone(extract_onset("ไม่แน่ใจค่ะ", asked=False))


class IntakeFlowTests(unittest.TestCase):
    def test_each_question_once_in_order(self):
        i = StructuredIntake()
        i.mark_asked("complaint")
        self.assertTrue(i.update("ปวดหัวค่ะ"))
        self.assertEqual(i.next_question("th"), ("severity", QUESTIONS["th"]["severity"]))
        self.assertTrue(i.update("7"))
        self.assertEqual(i.next_question("th"), ("onset", QUESTIONS["th"]["onset"]))
        self.assertTrue(i.update("ตั้งแต่เมื่อเช้า"))
        self.assertIsNone(i.next_question("th"))
        self.assertEqual(i.as_dict()["pain_score_0_10"], 7)

    def test_skips_questions_already_answered(self):
        i = StructuredIntake()
        i.mark_asked("complaint")
        i.update("I have a headache, 8 out of 10, since yesterday")
        self.assertIsNone(i.next_question("en"))

    def test_reask_once_then_move_on_never_loop(self):
        i = StructuredIntake()
        i.mark_asked("complaint")
        i.update("stomach ache")
        self.assertEqual(i.next_question("en")[1], QUESTIONS["en"]["severity"])
        self.assertFalse(i.update("hmm"))
        self.assertEqual(i.next_question("en")[1], REPROMPTS["en"]["severity"])
        self.assertFalse(i.update("blue"))
        self.assertEqual(i.next_question("en")[0], "onset")      # severity given up
        self.assertEqual(i.unusable_in_a_row, 2)
        self.assertIn("severity", i.as_dict()["not_answered"])
