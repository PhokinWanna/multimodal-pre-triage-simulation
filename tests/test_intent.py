import unittest

from triage_kiosk.intent import (NO, UNCLEAR, YES, classify_consent, find_wake_word,
                                 is_cancel_request, is_end_of_interview, is_ready_to_finish,
                                 choose_language, requested_language)


class ConsentTests(unittest.TestCase):
    def test_yes(self):
        for t in ["ยินยอมค่ะ", "ตกลงครับ", "ได้ค่ะ", "ไม่มีปัญหาค่ะ ยินยอมค่ะ", "ไม่มีปัญหาครับ", "โอเคค่ะ",
                  "Yes I agree", "Yes I agree, I know the policy", "okay", "sure, no problem", "I consent"]:
            self.assertEqual(classify_consent(t), YES, t)

    def test_no(self):
        for t in ["ไม่ยินยอมค่ะ", "ไม่ค่ะ", "ไม่ครับ", "ไม่ตกลง", "ปฏิเสธค่ะ", "ไม่ให้ค่ะ", "ไม่เอาค่ะ",
                  "no", "No thank you", "I don't agree", "I do not consent", "I disagree"]:
            self.assertEqual(classify_consent(t), NO, t)

    def test_unclear_is_never_consent(self):
        for t in ["", "   ", "อะไรนะคะ", "what?", "I know", "ขอฟังอีกครั้ง"]:
            self.assertEqual(classify_consent(t), UNCLEAR, t)


class CancelTests(unittest.TestCase):
    def test_cancel(self):
        for t in ["ยกเลิกค่ะ", "ไม่เอาแล้วครับ", "cancel", "I want to stop", "stop", "never mind, I'm leaving"]:
            self.assertTrue(is_cancel_request(t), t)

    def test_not_cancel(self):
        for t in ["I can't stop coughing since yesterday", "the pain won't stop", "it doesn't stop bleeding",
                  "I want to see the doctor", "enough pain to cry", "ปวดหัวค่ะ", ""]:
            self.assertFalse(is_cancel_request(t), t)


class EndOfInterviewTests(unittest.TestCase):
    def test_end(self):
        for t in ["หมดแล้วค่ะ", "แค่นี้ครับ", "ไม่มีแล้วค่ะ ขอบคุณค่ะ", "พอแล้วค่ะ", "ไม่มีค่ะ",
                  "that's all", "No, that's all, thank you", "nothing else", "I'm done", "nothing"]:
            self.assertTrue(is_end_of_interview(t), t)

    def test_not_end(self):
        for t in ["ยาความดันหมดแล้วค่ะ เวียนหัว", "ไม่มีไข้ แต่ปวดหัวมาก", "I haven't done anything",
                  "nothing helps the pain", "ปวดท้องค่ะ", ""]:
            self.assertFalse(is_end_of_interview(t), t)


class WakeWordTests(unittest.TestCase):
    TH, EN = ["สวัสดี", "ช่วยด้วย"], ["hello", "hi", "help"]

    def test_wake(self):
        self.assertEqual(find_wake_word("สวัสดีค่ะ", self.TH, self.EN), (True, "th"))
        self.assertEqual(find_wake_word("Hi there", self.TH, self.EN), (True, "en"))

    def test_no_false_wake(self):
        self.assertEqual(find_wake_word("this is the waiting room", self.TH, self.EN)[0], False)
        self.assertEqual(find_wake_word("which way to the pharmacy", self.TH, self.EN)[0], False)


class ConsentPoliteFormsTests(unittest.TestCase):
    def test_more_yes(self):
        for t in ["ยินดีค่ะ", "ค่ะ", "ครับ", "yup", "uh huh", "that's fine"]:
            self.assertEqual(classify_consent(t), YES, t)

    def test_negated_yes_words(self):
        for t in ["ไม่ยินดีค่ะ", "ไม่อนุญาตครับ", "ไม่ใช่ค่ะ"]:
            self.assertEqual(classify_consent(t), NO, t)


class ReadyToFinishTests(unittest.TestCase):
    def test_finish(self):
        for t in ["โอเค ไม่มีอะไรแล้วครับ", "โอเคพอแล้วครับ", "ไม่ค่ะ", "No.", "nope", "that's all thank you",
                  "อยากเจอพยาบาลครับ", "I want to see the nurse"]:
            self.assertTrue(is_ready_to_finish(t), t)

    def test_more_information_is_not_finish(self):
        for t in ["ผมเจ็บมาตั้งแต่ 6 โมงเช้า", "I also have a fever", "no, I also feel dizzy", "ไม่มีไข้แต่เวียนหัว"]:
            self.assertFalse(is_ready_to_finish(t), t)


class LanguageChoiceTests(unittest.TestCase):
    def test_explicit_request_wins(self):
        self.assertEqual(choose_language("English please", "th", 0.9, "th"), "en")
        self.assertEqual(choose_language("อิงลิชครับ", "th", 0.9, "th"), "en")     # Whisper's Thai spelling
        self.assertEqual(choose_language("ภาษาไทยครับ", "en", 0.9, "en"), "th")

    def test_spoken_language_used_when_confident(self):
        self.assertEqual(choose_language("Hello, I am fine", "en", 0.95, "th"), "en")
        self.assertEqual(choose_language("ค่ะ", "en", 0.55, "th"), "th")            # not sure -> keep
        self.assertEqual(choose_language("", "en", 0.99, "th"), "th")

    def test_mid_interview_request(self):
        self.assertEqual(requested_language("can you speak English please"), "en")
        self.assertIsNone(requested_language("ปวดหัวค่ะ"))
