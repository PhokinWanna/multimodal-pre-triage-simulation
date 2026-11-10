"""Runs scripts/checks/kiosk_fsm_sim.py (real state machine with fake hardware) as a test."""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KioskFsmSimulationTest(unittest.TestCase):
    def test_all_scenarios_behave_safely(self):
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "checks", "kiosk_fsm_sim.py")],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-2000:])
