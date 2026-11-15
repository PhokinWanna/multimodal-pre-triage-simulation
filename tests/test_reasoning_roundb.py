"""Round-B reasoning: prompt versions, step-derived level, weighted fusion floor, decision record."""
import unittest

from triage_kiosk.triage import prompts
from triage_kiosk.triage.reasoning import ReasoningEngine

CHEST = {"chest_clutching": True, "abdominal_clutching": False, "head_clutching": False}
NOTHING = {"chest_clutching": False, "abdominal_clutching": False, "head_clutching": False}
SOAP = {"urgency": "x", "subjective": "s", "objective": "o", "assessment": "a", "plan": "p"}


def engine_answering(out, reason=None):
    eng = ReasoningEngine.__new__(ReasoningEngine)
    eng.endpoint = "http://x"
    eng.prompts_seen = []

    def fake(prompt, temperature, seed, timeout, num_predict):
        eng.prompts_seen.append((prompt, num_predict))
        return (None if reason else dict(out)), reason, (None if reason else 1.0)
    eng._call_llm = fake
    return eng


def case(eng, text="mild cough", signs=NOTHING, skin="NORMAL", **kw):
    conv = [{"role": "patient", "text": text}] if text else []
    return eng.evaluate_final_case(conv, signs, skin, "NORMAL_COMPLETE", "en", **kw)


class PromptTests(unittest.TestCase):
    def test_v2_prompt_carries_steps_camera_block_and_weights(self):
        eng = engine_answering({"esi_level": 4, **SOAP})
        case(eng, signs=CHEST, skin="UNKNOWN", skin_reason="face too small in the image",
             prompt_version="v2", fusion_mode="weighted", weights=(0.6, 0.4))
        prompt, num_predict = eng.prompts_seen[0]
        for piece in ("STEP A", "STEP B", "STEP C", "hand held on the chest",
                      "not assessed (face too small in the image)",
                      "Combined visual risk: HIGH (score 0.60 of 1.00; posture weight 60%, skin weight 40%)"):
            self.assertIn(piece, prompt)
        self.assertEqual(num_predict, prompts.NUM_PREDICT["v2"])
        self.assertNotIn("EXAMPLES", prompt)

    def test_v3_adds_examples_and_v1_keeps_the_old_layout(self):
        eng = engine_answering({"esi_level": 4, **SOAP})
        case(eng, prompt_version="v3", fusion_mode="weighted")
        case(eng, prompt_version="v1", fusion_mode="independent")
        self.assertIn("EXAMPLES", eng.prompts_seen[0][0])
        self.assertIn("[SUSTAINED PERCEPTION TELEMETRY]", eng.prompts_seen[1][0])
        self.assertNotIn("STEP A", eng.prompts_seen[1][0])

    def test_independent_mode_shows_no_combined_risk(self):
        eng = engine_answering({"esi_level": 4, **SOAP})
        case(eng, signs=CHEST, prompt_version="v2", fusion_mode="independent")
        self.assertNotIn("Combined visual risk:", eng.prompts_seen[0][0].split("[CAMERA]")[-1])

    def test_no_words_and_switched_off_parts(self):
        eng = engine_answering({"esi_level": 5, **SOAP})
        v = case(eng, text="", signs=None, skin="NOT_ASSESSED", prompt_version="v2", fusion_mode="weighted")
        user_part = eng.prompts_seen[0][0].split("[PATIENT'S WORDS]")[-1]
        self.assertIn(prompts.NO_WORDS, user_part)
        self.assertIn("Posture: not assessed", user_part)
        self.assertIn("Combined visual risk: not assessed", user_part)
        self.assertEqual((v["decision"]["posture_assessed"], v["decision"]["visual"]["band"]), (False, "NOT_ASSESSED"))


class StepTests(unittest.TestCase):
    def test_level_from_steps(self):
        f = prompts.esi_from_steps
        self.assertEqual(f({"step_a": "yes", "step_b": "no", "resources": 0}), 1)
        self.assertEqual(f({"step_a": "no", "step_b": "YES", "resources": 0}), 2)
        self.assertEqual([f({"step_a": "no", "step_b": "no", "resources": r}) for r in (3, 2, 1, 0, "1")], [3, 3, 4, 5, 4])
        self.assertIsNone(f({"step_a": "no", "step_b": "perhaps", "resources": 1}))
        self.assertIsNone(f({"step_a": "no", "step_b": "no", "resources": "many"}))

    def test_choice_between_level_field_and_steps(self):
        out = {"step_a": "no", "step_b": "no", "resources": 0, "esi_level": 3, **SOAP}
        v_level = case(engine_answering(out), prompt_version="v2", fusion_mode="weighted", esi_from_steps=False)
        v_steps = case(engine_answering(out), prompt_version="v2", fusion_mode="weighted", esi_from_steps=True)
        self.assertEqual((v_level["esi_level"], v_steps["esi_level"]), (3, 5))
        d = v_steps["decision"]
        self.assertEqual((d["esi_llm_level_field"], d["esi_llm_from_steps"], d["esi_llm"]), (3, 5, 5))
        self.assertEqual(v_steps["urgency"], "Non-urgent")

    def test_usable_steps_rescue_an_invalid_level_field(self):
        out = {"step_a": "no", "step_b": "yes", "resources": 2, "esi_level": "high", **SOAP}
        v = case(engine_answering(out), prompt_version="v2", fusion_mode="weighted", esi_from_steps=False)
        self.assertEqual((v["esi_level"], v["decision"]["llm_valid"]), (2, True))

    def test_nothing_usable_falls_back_to_rules(self):
        v = case(engine_answering({"esi_level": 9}), prompt_version="v2", fusion_mode="weighted")
        self.assertFalse(v["decision"]["llm_valid"])
        self.assertIn("INVALID_LLM_VERDICT", v["audit_mode"])
        self.assertEqual(v["esi_level"], v["decision"]["esi_rules"])


class WeightedFloorTests(unittest.TestCase):
    LLM4 = {"step_a": "no", "step_b": "no", "resources": 1, "esi_level": 4, **SOAP}

    def final(self, weights, signs, skin, prob, policy="high_only"):
        v = case(engine_answering(self.LLM4), signs=signs, skin=skin, skin_probability=prob,
                 prompt_version="v2", fusion_mode="weighted", weights=weights, cap_policy=policy)
        return v["esi_level"], v["decision"]

    def test_chest_alone_is_capped_at_50_and_60_but_not_at_40(self):
        self.assertEqual([self.final(w, CHEST, "UNKNOWN", None)[0] for w in ((0.5, 0.5), (0.6, 0.4), (0.4, 0.6))],
                         [2, 2, 4])

    def test_flushing_alone_is_capped_only_when_skin_has_60(self):
        self.assertEqual([self.final(w, NOTHING, "RED_FLUSHING", 0.9)[0] for w in ((0.5, 0.5), (0.6, 0.4), (0.4, 0.6))],
                         [4, 4, 2])

    def test_moderate_band_caps_only_under_the_strict_policy(self):
        self.assertEqual(self.final((0.5, 0.5), NOTHING, "RED_FLUSHING", 0.9, "high_and_moderate")[0], 3)
        self.assertEqual(self.final((0.5, 0.5), CHEST, "UNKNOWN", None, "none")[0], 4)

    def test_decision_record_lets_the_floor_be_recomputed(self):
        final, d = self.final((0.5, 0.5), CHEST, "NORMAL", 0.1)
        self.assertEqual((d["esi_llm"], d["esi_before_floor"], d["esi_final"], final), (4, 4, 2, 2))
        self.assertEqual([(c["cap"], c["source"]) for c in d["caps"]], [(2, "visual_band")])
        self.assertEqual(d["visual"]["band"], "HIGH")

    def test_independent_mode_keeps_the_old_separate_rules(self):
        v = case(engine_answering(self.LLM4), signs=NOTHING, skin="RED_FLUSHING", skin_probability=0.9,
                 prompt_version="v2", fusion_mode="independent")
        self.assertEqual((v["esi_level"], v["decision"]["caps"][0]["source"], v["decision"]["visual"]), (3, "skin", None))

    def test_pain_score_floor_and_esi1_scope(self):
        v = case(engine_answering(self.LLM4), prompt_version="v2", fusion_mode="weighted",
                 intake={"complaint": "x", "pain_score_0_10": 8})
        self.assertEqual((v["esi_level"], v["scope"]), (3, "IN_SCOPE"))
        v = case(engine_answering({"step_a": "yes", "step_b": "no", "resources": 2, "esi_level": 1, **SOAP}),
                 prompt_version="v2", fusion_mode="weighted")
        self.assertEqual((v["esi_level"], v["scope"]), (1, "OUT_OF_SCOPE_IMMEDIATE_STAFF"))
        self.assertIn("OUT OF KIOSK SCOPE", v["urgency"])


class RulesOnlyTests(unittest.TestCase):
    def rules(self, text, signs, skin, prob, weights=(0.5, 0.5), mode="weighted"):
        eng = engine_answering({})
        v = eng.evaluate_final_case([{"role": "patient", "text": text}], signs, skin, "NORMAL_COMPLETE", "en",
                                    use_llm=False, skin_probability=prob, fusion_mode=mode, weights=weights,
                                    prompt_version="v2")
        self.assertEqual(eng.prompts_seen, [])
        return v["esi_level"]

    def test_weighted_rules(self):
        self.assertEqual(self.rules("I need a refill", NOTHING, "UNKNOWN", None), 5)       # unknown skin adds nothing
        self.assertEqual(self.rules("my knee hurts", NOTHING, "NORMAL", 0.1), 4)
        self.assertEqual(self.rules("my knee hurts", NOTHING, "RED_FLUSHING", 0.9), 3)     # MODERATE -> 3 without an LLM
        self.assertEqual(self.rules("my knee hurts", CHEST, "NORMAL", 0.1), 2)
        self.assertEqual(self.rules("my knee hurts", CHEST, "NORMAL", 0.1, (0.4, 0.6)), 3)
        self.assertEqual(self.rules("he is unconscious", NOTHING, "NORMAL", 0.1), 1)

    def test_independent_rules_unchanged(self):
        self.assertEqual(self.rules("my knee hurts", CHEST, "NORMAL", 0.1, mode="independent"), 2)
        self.assertEqual(self.rules("my knee hurts", NOTHING, "NORMAL", 0.1, mode="independent"), 4)


if __name__ == "__main__":
    unittest.main()


class RetryTests(unittest.TestCase):
    def test_cut_off_json_is_retried_once_with_a_larger_budget(self):
        eng = ReasoningEngine.__new__(ReasoningEngine)
        eng.endpoint = "http://x"
        budgets = []

        def once(prompt, temperature, seed, timeout, num_predict):
            budgets.append(num_predict)
            if len(budgets) == 1:
                return None, "DETERMINISTIC_FALLBACK_JSON_PARSE_ERROR", None
            return {"esi_level": 3, **SOAP}, None, 2.0
        eng._call_llm_once = once
        v = case(eng, prompt_version="v1", fusion_mode="weighted")
        self.assertEqual((budgets, v["esi_level"], v["decision"]["llm_valid"]), ([200, 400], 3, True))

    def test_no_second_retry(self):
        eng = ReasoningEngine.__new__(ReasoningEngine)
        eng.endpoint = "http://x"
        calls = []
        eng._call_llm_once = lambda *a: calls.append(a) or (None, "DETERMINISTIC_FALLBACK_JSON_PARSE_ERROR", None)
        v = case(eng, prompt_version="v1", fusion_mode="weighted")
        self.assertEqual((len(calls), v["decision"]["llm_valid"]), (2, False))
