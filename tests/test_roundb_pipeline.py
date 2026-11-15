"""Dry run of the round-B scripts with a fake camera and a fake LLM (no Ollama, no photos)."""

import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "m3"))
import m3_common  # noqa: E402
import prompt_dev_eval  # noqa: E402
import roundb_common  # noqa: E402
import run_m3  # noqa: E402
import run_roundB  # noqa: E402

REF = {"S01": 1, "S02": 2, "S03": 2, "S04": 3, "S05": 3, "S06": 4, "S07": 4, "S08": 5, "S09": 5, "S10": 3}
CATS = {"S01": "both_chest", "S02": "chest", "S03": "flush", "S04": "abdomen", "S05": "both_head",
        "S06": "head", "S07": "neither", "S08": "neither", "S09": "flush", "S10": "both_abdomen"}
SOAP = {"urgency": "x", "subjective": "s", "objective": "o", "assessment": "a", "plan": "p"}


def steps_for(level):
    return {"step_a": "yes" if level == 1 else "no", "step_b": "yes" if level == 2 else "no",
            "resources": {3: 2, 4: 1, 5: 0}.get(level, 2)}


def fake_analyse(vision, path):
    stem = path.stem
    role = stem.rsplit("_", 1)[-1] if stem.endswith(("_body", "_face")) else "single"
    cat = stem.rsplit("_", 2 if role != "single" else 1)[0]
    posture = cat.replace("both_", "") if cat not in ("flush", "neither") and role != "face" else "none"
    red = cat == "flush" or role == "face"
    small = cat == "neither"                               # the image check refuses these faces
    return {"person_found": True, "face_found": True,
            "visual_signs": {"chest_clutching": posture == "chest", "abdominal_clutching": posture == "abdomen",
                             "head_clutching": posture == "head"},
            "posture_label": posture, "posture_filters": [],
            "skin_status": "UNKNOWN" if small else "RED_FLUSHING" if red else "NORMAL",
            "skin_probability": None if small else 0.9 if red else 0.1,
            "skin_reason": "face too small in the image" if small else "",
            "skin_patches": {}, "skin_quality": {}, "skin_method": "patch_cnn_qc"}


class RoundBDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "images" / "test").mkdir(parents=True)
        rows = []
        for sid, cat in CATS.items():
            posture = cat.replace("both_", "") if cat not in ("flush", "neither") else "none"
            if cat.startswith("both_"):
                for role, p, fl in (("body", posture, "no"), ("face", "none", "yes")):
                    f = f"{cat}_{sid}_{role}.jpg"
                    (root / "images" / "test" / f).write_bytes(b"x")
                    rows.append({"file": f, "set": "test", "category": cat, "true_posture": p, "true_flushing": fl})
                continue
            f = f"{cat}_{sid}.jpg"
            (root / "images" / "test" / f).write_bytes(b"x")
            rows.append({"file": f, "set": "test", "category": cat, "true_posture": posture,
                         "true_flushing": "yes" if cat == "flush" else "no"})
        self.sources = root / "image_sources.csv"
        m3_common.write_csv(self.sources, rows, m3_common.SOURCE_COLUMNS)
        self.scen = root / "scenarios.csv"
        m3_common.write_csv(self.scen, [{"id": s, "image_category": c, "image_file": "", "language": "en",
                                         "patient_text": f"my problem is number {s}"} for s, c in CATS.items()],
                            ["id", "image_category", "image_file", "language", "patient_text"])
        self.labels = root / "labels_final.csv"
        m3_common.write_csv(self.labels, [{"id": s, "esi_reference": e} for s, e in REF.items()], ["id", "esi_reference"])
        self.dev = root / "dev_scenarios.csv"
        m3_common.write_csv(self.dev, [
            {"id": "D01", "esi_reference": 2, "language": "en", "patient_text": "dev case two", "camera_posture": "chest",
             "camera_skin_status": "NORMAL", "camera_skin_probability": "0.1"},
            {"id": "D02", "esi_reference": 5, "language": "en", "patient_text": "dev case five", "camera_posture": "none",
             "camera_skin_status": "RED_FLUSHING", "camera_skin_probability": "0.8"},
            {"id": "D03", "esi_reference": 4, "language": "en", "patient_text": "dev case four", "camera_posture": "none",
             "camera_skin_status": "UNKNOWN", "camera_skin_probability": ""}],
            ["id", "esi_reference", "language", "patient_text", "camera_posture", "camera_skin_status", "camera_skin_probability"])
        self.root = root
        self._saved = (m3_common.IMAGES_DIR, m3_common.SOURCES_CSV, run_m3.IMAGES_DIR, run_roundB.IMAGES_DIR,
                       run_roundB.analyse_photo, prompt_dev_eval.SELECTION_JSON, prompt_dev_eval.ROUNDB_DIR,
                       prompt_dev_eval.RUN_DIR, prompt_dev_eval.DEV_SCENARIOS_CSV)
        m3_common.IMAGES_DIR = run_m3.IMAGES_DIR = run_roundB.IMAGES_DIR = root / "images"
        m3_common.SOURCES_CSV = self.sources
        run_roundB.analyse_photo = fake_analyse
        prompt_dev_eval.SELECTION_JSON = root / "roundB" / "prompt_selection.json"
        prompt_dev_eval.ROUNDB_DIR = root / "roundB"
        prompt_dev_eval.RUN_DIR = root / "roundB" / "run_main"
        prompt_dev_eval.DEV_SCENARIOS_CSV = self.dev
        self.calls = []

    def tearDown(self):
        (m3_common.IMAGES_DIR, m3_common.SOURCES_CSV, run_m3.IMAGES_DIR, run_roundB.IMAGES_DIR, run_roundB.analyse_photo,
         prompt_dev_eval.SELECTION_JSON, prompt_dev_eval.ROUNDB_DIR, prompt_dev_eval.RUN_DIR,
         prompt_dev_eval.DEV_SCENARIOS_CSV) = self._saved
        self.tmp.cleanup()

    # ------------------------------------------------------------------ fake LLMs
    def dev_llm(self, prompt, temperature, seed, timeout, num_predict):
        """v1 always says 3; v2/v3: the level field is one too urgent, the step answers are right."""
        self.calls.append(prompt)
        user = prompt.split("<|start_header_id|>user")[-1]
        level = {"two": 2, "five": 5, "four": 4}[re.search(r"dev case (\w+)", user).group(1)]
        if "STEP A" not in prompt:
            return {"esi_level": 3, **SOAP}, None, 1.0
        return {**steps_for(level), "esi_level": max(1, level - 1), **SOAP}, None, 2.0

    def fake_test_llm(self, prompt, temperature, seed, timeout, num_predict):
        self.calls.append((prompt, temperature, seed))
        user = prompt.split("<|start_header_id|>user")[-1]
        m = re.search(r"number (S\d+)", user)
        if not m:                                           # patient's words switched off
            level = 2 if "Combined visual risk: HIGH" in user else 4
        else:
            sid = m.group(1)
            level = REF[sid] + (1 if sid in ("S03", "S06") else 0)      # two under-triage errors by the LLM
            if sid == "S08":
                return {"esi_level": "??"}, None, 1.0                   # one unusable output
        return {**steps_for(level), "esi_level": level, **SOAP}, None, 1.5

    # ------------------------------------------------------------------ tests
    def test_prompt_selection_follows_the_rule_and_is_not_redone(self):
        self.assertEqual(prompt_dev_eval.main([], llm=self.dev_llm), 0)
        sel = json.loads(prompt_dev_eval.SELECTION_JSON.read_text(encoding="utf-8"))
        table = {t["candidate"]: t for t in sel["table"]}
        self.assertEqual(sel["selected"], {"candidate": "v2-steps", "prompt_version": "v2", "esi_from_steps": True})
        self.assertEqual(table["v2-steps"]["accuracy"], 1.0)
        self.assertEqual(table["v2-steps"]["kappa_linear"], table["v3-steps"]["kappa_linear"])   # tie -> simpler one
        self.assertLess(table["v1"]["kappa_linear"], table["v2-steps"]["kappa_linear"])
        self.assertEqual(len(self.calls), 9)
        self.calls.clear()
        self.assertEqual(prompt_dev_eval.main([], llm=self.dev_llm), 0)          # second call: nothing is redone
        self.assertEqual(self.calls, [])

    def run_test(self, extra=()):
        args = run_roundB.parse_args(["--scenarios", str(self.scen), *extra])
        return run_roundB.run(args, llm=self.fake_test_llm, vision=object(), out_dir=self.root / "roundB" / "run_main",
                              selection={"candidate": "v2-steps", "prompt_version": "v2", "esi_from_steps": True})

    def test_run_records_every_configuration_and_resumes_from_the_cache(self):
        self.assertEqual(self.run_test(), 0)
        out = self.root / "roundB" / "run_main"
        recs = [json.loads(x) for x in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(recs), 10 * (len(run_roundB.LLM_CONFIGS) + len(run_roundB.RULES_CONFIGS)))
        by = {(r["config"], r["id"]): r for r in recs}
        self.assertTrue(all(t == 0.0 and s == 0 for _, t, s in self.calls))
        # identical prompts (words off, same camera result) are asked once and then read from the cache
        self.assertLess(len(self.calls), 10 * len(run_roundB.LLM_CONFIGS))
        # chest alone, normal skin 0.1: HIGH at 50/50 and 60/40, MODERATE at 40/60
        self.assertEqual([by[(c, "S02")]["visual_band"] for c in ("W50", "W60", "W40")], ["HIGH", "HIGH", "MODERATE"])
        # flushing alone 0.9: the LLM's under-triage (3) is corrected only when skin has 60 %
        self.assertEqual([by[(c, "S03")]["esi_final"] for c in ("W50", "W60", "W40")], [3, 3, 2])
        self.assertEqual(by[("W40", "S03")]["esi_llm"], 3)
        # switched-off parts are not sent
        self.assertIsNone(by[("T", "S02")]["visual_score"])
        self.assertEqual(by[("TP", "S03")]["visual_band"], "LOW")
        self.assertEqual((by[("IND", "S03")]["fusion_mode"], by[("IND", "S03")]["visual_band"]), ("independent", None))
        # unusable LLM output -> rules decide, and it is recorded
        self.assertEqual((by[("W50", "S08")]["llm_valid"], by[("W50", "S08")]["esi_final"]),
                         (False, by[("W50", "S08")]["esi_rules"]))
        # rules-only configurations never call the LLM
        self.assertTrue(all(not r["llm_used"] and r["esi_llm"] is None for r in recs if r["config"].endswith("_rules")))
        self.assertEqual(by[("W50", "S01")]["scope"], "OUT_OF_SCOPE_IMMEDIATE_STAFF")

        n_first = len(self.calls)
        self.calls.clear()
        self.assertEqual(self.run_test(), 0)                                    # same command again
        self.assertEqual(self.calls, [], "a finished run must be read back from the cache")
        self.assertGreater(n_first, 0)

    def test_no_qc_sensitivity_uses_the_other_skin_result(self):
        seen = []
        orig = run_roundB.analyse_photo

        def analyse(vision, path):
            seen.append(getattr(vision, "skin_pipeline", "?"))
            return fake_analyse(vision, path)
        run_roundB.analyse_photo = analyse
        try:
            self.assertEqual(self.run_test(["--no-qc"]), 0)
        finally:
            run_roundB.analyse_photo = orig
        out = self.root / "roundB" / "run_main"
        self.assertTrue((out / "vision_noQC.json").is_file())
        recs = [json.loads(x) for x in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        names = {r["config"] for r in recs}
        self.assertTrue({"W50_noQC", "W60_noQC", "W40_noQC", "TS_noQC", "S_noQC", "PS_noQC", "W50_rules_noQC"} <= names)
        self.assertNotIn("T_noQC", names)

    def test_a_folder_from_another_selection_is_never_mixed(self):
        self.assertEqual(self.run_test(), 0)
        args = run_roundB.parse_args(["--scenarios", str(self.scen)])
        rc = run_roundB.run(args, llm=self.fake_test_llm, vision=object(), out_dir=self.root / "roundB" / "run_main",
                            selection={"candidate": "v1", "prompt_version": "v1", "esi_from_steps": False})
        self.assertEqual(rc, 1)

    def test_analysis_tables_and_contribution(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        import analyze_roundB
        self.assertEqual(self.run_test(["--consistency", "--no-qc"]), 0)
        out = self.root / "analysis"
        rc = analyze_roundB.main(["--run", str(self.root / "roundB" / "run_main"), "--labels", str(self.labels),
                                  "--out", str(out), "--no-round-a"])
        self.assertEqual(rc, 0)
        res = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((res["n"], res["n_in_scope"], res["n_out_of_scope"]), (10, 9, 1))
        main = res["systems"]["B_main"]
        self.assertEqual(main["out_of_scope"]["detected_strict"]["value"], 1.0)
        self.assertLessEqual(main["in_scope"]["under_triage"]["value"], res["systems"]["B_llm_only"]["in_scope"]["under_triage"]["value"])
        share = res["contribution"]["llm_only"]["share"]
        self.assertAlmostEqual(sum(share.values()), 1.0, places=6)
        self.assertGreater(share["text"], share["skin"])                         # the fake LLM mostly follows the words
        self.assertIn("W50:high_and_moderate", res["floor_policies"])
        self.assertEqual(res["consistency"]["runs"], 10)
        for name in ("report.md", "per_case.csv", "fig_confusion.png", "fig_weights.png", "fig_weights_rules.png",
                     "fig_modalities.png", "fig_contribution.png", "fig_rounds.png", "fig_recall.png",
                     "fig_weights_noQC.png", "fig_contribution_noQC.png"):
            self.assertTrue((out / name).is_file(), name)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("weights_noQC", res)
        self.assertAlmostEqual(sum(res["contribution_noQC"]["llm_only"]["share"].values()), 1.0, places=6)
        for heading in ("In-scope scenarios", "Out-of-scope scenarios", "Posture : skin weights", "How much each input",
                        "skin part always answers"):
            self.assertIn(heading, report)

    def test_shapley_is_exact_for_an_additive_game(self):
        import analyze_roundB as A
        vals = {c: 1.0 * ("text" in c) + 0.5 * ("posture" in c) for c in A.COALITION_CONFIG}
        self.assertEqual(A.shapley(vals), {"text": 1.0, "posture": 0.5, "skin": 0.0})


if __name__ == "__main__":
    unittest.main()
