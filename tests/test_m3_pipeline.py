"""End-to-end dry run of the M3 scripts with a fake camera and a fake LLM (no Ollama, no photos)."""

import csv
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "m3"))
import m3_common  # noqa: E402
import run_m3  # noqa: E402

REF = {"S01": 1, "S02": 2, "S03": 2, "S04": 3, "S05": 3, "S06": 4, "S07": 4, "S08": 5, "S09": 5, "S10": 3}
CATS = {"S01": "both_chest", "S02": "chest", "S03": "flush", "S04": "abdomen", "S05": "both_head",
        "S06": "head", "S07": "neither", "S08": "neither", "S09": "flush", "S10": "both_abdomen"}


class FakeVision:
    posture_version, skin_pipeline, skin_model = "v2", "patch", None

    def analyze_image(self, img):
        return self.next_result


class M3DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "images" / "test").mkdir(parents=True)
        self.sources = root / "image_sources.csv"
        rows = []
        for sid, cat in CATS.items():
            posture = cat.replace("both_", "") if cat not in ("flush", "neither") else "none"
            if cat.startswith("both_"):   # two photos: body (posture, normal skin) + flushed face
                for role, p, fl in (("body", posture, "no"), ("face", "none", "yes")):
                    f = f"{cat}_{sid}_{role}.jpg"
                    (root / "images" / "test" / f).write_bytes(b"x")
                    rows.append({"file": f, "set": "test", "category": cat, "true_posture": p, "true_flushing": fl})
                continue
            f = f"{cat}_{sid}.jpg"
            (root / "images" / "test" / f).write_bytes(b"x")
            rows.append({"file": f, "set": "test", "category": cat, "true_posture": posture,
                         "true_flushing": "yes" if cat == "flush" else "no"})
        m3_common.write_csv(self.sources, rows, m3_common.SOURCE_COLUMNS)
        self.scen = root / "scenarios.csv"
        m3_common.write_csv(self.scen, [{"id": s, "image_category": c, "image_file": "", "language": "en",
                                         "patient_text": f"text {s}"} for s, c in CATS.items()],
                            ["id", "image_category", "image_file", "language", "patient_text"])
        self.labels = root / "labels_final.csv"
        m3_common.write_csv(self.labels, [{"id": s, "esi_reference": e} for s, e in REF.items()], ["id", "esi_reference"])
        self.root = root
        self._saved = (m3_common.IMAGES_DIR, m3_common.SOURCES_CSV, run_m3.IMAGES_DIR)
        m3_common.IMAGES_DIR = run_m3.IMAGES_DIR = root / "images"
        m3_common.SOURCES_CSV = self.sources
        self.calls = []

    def tearDown(self):
        m3_common.IMAGES_DIR, m3_common.SOURCES_CSV, run_m3.IMAGES_DIR = self._saved
        self.tmp.cleanup()

    def fake_llm(self, conversation, signs, skin, exit_reason, lang, temperature, seed, timeout):
        sid = conversation[0]["text"].split()[-1]
        self.calls.append((sid, temperature, seed, timeout, signs, skin))
        esi = REF[sid] if sid not in ("S03", "S06") else REF[sid] + 1     # two deliberate under-triage errors
        if temperature > 0 and seed % 5 == 0:
            esi = max(1, esi - 1)                                           # some sampling variation in B
        return {"esi_level": esi, "urgency": "x", "audit_mode": "LLM_INFERENCE_VALIDATED", "inference_latency_sec": 1.2}

    def run_mode(self, mode):
        import cv2  # noqa: F401  (run_m3 reads photos with OpenCV)
        vision = FakeVision()
        orig = run_m3.analyse_photo

        def fake_analyse(v, path):
            stem = path.stem
            role = stem.rsplit("_", 1)[-1] if stem.endswith(("_body", "_face")) else "single"
            cat = stem.rsplit("_", 2 if role != "single" else 1)[0]
            posture = cat.replace("both_", "") if cat not in ("flush", "neither") and role != "face" else "none"
            red = cat == "flush" or role == "face"       # both_* body photos have NORMAL skin on purpose
            return {"person_found": True, "face_found": True,
                    "visual_signs": {"chest_clutching": posture == "chest", "abdominal_clutching": posture == "abdomen",
                                     "head_clutching": posture == "head"},
                    "posture_label": posture, "posture_filters": [], "posture_debug": {},
                    "skin_status": "RED_FLUSHING" if red else "NORMAL",
                    "skin_probability": 0.9 if red else 0.1, "skin_patches": {}, "skin_method": "patch_cnn"}
        run_m3.analyse_photo = fake_analyse
        try:
            args = run_m3.parse_args(["--mode", mode, "--scenarios", str(self.scen), "--out", str(self.root / "runs")])
            return run_m3.run(args, llm_call=self.fake_llm, vision=vision)
        finally:
            run_m3.analyse_photo = orig

    def test_replay_records_every_call_with_pre_registered_settings(self):
        out_a = self.run_mode("A")
        lines = [json.loads(x) for x in (out_a / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 10)
        self.assertTrue(all(c[1] == 0.0 and c[2] == 0 and c[3] == 60.0 for c in self.calls))
        cfg = json.loads((out_a / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["temperature"], cfg["runs_per_scenario"], cfg["llm_timeout_s"]), (0.0, 1, 60.0))
        vis = {r["id"]: r for r in csv.DictReader(open(out_a / "vision.csv", encoding="utf-8"))}
        self.assertEqual(set(vis), set(REF))
        # two-photo case: posture from the body photo, skin from the separate face photo
        self.assertTrue(vis["S05"]["image_file"].endswith("_body.jpg") and vis["S05"]["image_file_skin"].endswith("_face.jpg"))
        self.assertEqual((vis["S05"]["posture_label"], vis["S05"]["skin_status"]), ("head", "RED_FLUSHING"))
        self.assertEqual(vis["S03"]["image_file_skin"], "")                       # single-photo case
        sent = {c[0]: (c[4], c[5]) for c in self.calls}
        self.assertEqual(sent["S01"], ({"chest_clutching": True, "abdominal_clutching": False, "head_clutching": False},
                                       "RED_FLUSHING"))

        self.calls.clear()
        out_b = self.run_mode("B")
        self.assertEqual(len((out_b / "cases.jsonl").read_text(encoding="utf-8").splitlines()), 100)
        self.assertEqual(sorted({c[2] for c in self.calls}), list(range(1, 11)))
        self.assertTrue(all(c[1] == 0.8 for c in self.calls))

    def test_analysis_produces_tables_and_figures(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        import analyze_m3
        out_a, out_b = self.run_mode("A"), self.run_mode("B")
        saved = analyze_m3.RESULTS_DIR
        analyze_m3.RESULTS_DIR = self.root / "results"
        argv = sys.argv
        sys.argv = ["analyze_m3", "--run-a", str(out_a), "--run-b", str(out_b), "--labels", str(self.labels),
                    "--name", "dry"]
        try:
            self.assertEqual(analyze_m3.main(), 0)
        finally:
            sys.argv, analyze_m3.RESULTS_DIR = argv, saved
        res_dir = self.root / "results" / "analysis_dry"
        for f in ("metrics_table.csv", "metrics_table.md", "per_case.csv", "camera_components.md", "summary.json",
                  "confusion_matrix.png", "confusion_matrix.pdf", "recall_per_level.png", "roc.png"):
            self.assertTrue((res_dir / f).is_file(), f)
        summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(summary["option_A"]["accuracy"]["value"], 0.8)          # 8 of 10 correct
        self.assertAlmostEqual(summary["option_A"]["under_triage"]["value"], 0.2)      # the two deliberate errors
        self.assertEqual(summary["errors_by_source_A"], {"reasoning": 2})              # camera was right both times
        self.assertEqual(summary["camera"], {"posture_accuracy": 1.0, "skin_accuracy": 1.0})


if __name__ == "__main__":
    unittest.main()
