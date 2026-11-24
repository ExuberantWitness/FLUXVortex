"""Leading-edge pin sign contract: the pinned LE row must carry the ATTACHED sign.

Why this test exists
--------------------
On the legacy separated path the leading-edge row of each separated strip is replaced by a
pinned value.  Two LESP sign conventions meet there: the 2-D source bank returns
``A0 = +lesp_crit`` on a shed strip (Ramesh: LESP == A0), while the 3-D side defines
``lesp_pre = -gamma_pre/scale`` and then pins ``Gamma_LE = -A0*scale``.  The 3-D minus sign
is therefore applied to a quantity that never carried it, and the pinned row comes out with
the OPPOSITE sign to the attached circulation it is meant to bound.  Measured at runtime on
A16 (30/30 strips): attached ``gamma_pre = +0.004``, pinned ``Gamma_LE < 0``.

The separation mask is ``|LESP| > crit`` (sign-symmetric), so this never changed which
strips separate -- only what they are pinned to.  That is exactly why it survived: every
existing test is sign-blind.

What is asserted
----------------
On every separated strip, ``sign(Gamma_LE pinned) == sign(gamma_pre attached)`` for the
whole run.  No exception is made for a strip whose ``gamma_pre`` is ~0: at positive
incidence on this case all strips carry a clearly positive attached circulation, so a
near-zero or negative pinned row is a failure, not a tolerance question.

Expected per arm
----------------
    legacy       FAIL   (the behaviour under study)
    signed       pass
    signed_mag   pass
    joint        pass   (its target is -sign(lesp_pre)*crit*scale with lesp_pre=-G/scale)
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = "ROJ11-A16"
STEPS = 12


def _run_and_capture(env_overrides: dict[str, str]) -> dict[str, object]:
    """Run the case in a SUBPROCESS with the given env and return per-step sign data.

    A subprocess is required because FLUXV_LE_PIN / FLUXV_JOINT_SEPARATION are read at
    import time and must not leak between arms.
    """

    code = r"""
import json, sys
import torch
from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

out = {"steps": [], "cn": []}
state = {"n": 0}

def grab(frame, event, arg):
    if (event == "return"
            and frame.f_code.co_name == "propose"
            and frame.f_code.co_filename.endswith("q16_flux_v5m_native.py")):
        loc = frame.f_locals
        if "gamma_pre" in loc and "gamma" in loc and "scale" in loc:
            ns = int(loc["scale"].shape[0])
            gp = loc["gamma_pre"][:ns].detach().float().cpu()
            g = loc["gamma"][:ns].detach().float().cpu()
            sep = loc.get("surface_separated")
            sep = sep.detach().cpu() if sep is not None else torch.ones(ns, dtype=torch.bool)
            out["steps"].append({
                "pre_sign": torch.sign(gp).tolist(),
                "pinned_sign": torch.sign(g).tolist(),
                "separated": [bool(v) for v in sep],
                "pre_abs_min": float(gp.abs().min()),
            })
    return grab

runner = RojratsirikulCaseRunner(ROJRATSIRIKUL2011_UNIFIED_CASES["ROJ11-A16"], device="cuda:0")
runner.build()
sys.setprofile(grab)
payload = runner.run(max_aero_steps=STEPS)
sys.setprofile(None)
out["cn"] = [float(r["cn"]) for r in payload["records"]]
print("CAPTURE=" + json.dumps(out))
""".replace("STEPS", str(STEPS))
    environment = dict(os.environ)
    environment.update(env_overrides)
    environment["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "platform"),
         os.path.join(REPO_ROOT, "platform", "warp_vpm"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=environment, cwd=REPO_ROOT,
    )
    marker = [line for line in completed.stdout.splitlines() if line.startswith("CAPTURE=")]
    assert completed.returncode == 0 and marker, (
        f"arm {env_overrides} did not produce a capture\n"
        f"stdout tail:\n{completed.stdout[-800:]}\nstderr tail:\n{completed.stderr[-1500:]}"
    )
    import json

    return json.loads(marker[-1][len("CAPTURE="):])


def _first_mismatch(capture: dict[str, object]) -> tuple[int, int] | None:
    """(step index, strip) of the first separated strip whose pinned sign differs."""

    for step_index, step in enumerate(capture["steps"]):
        for strip, separated in enumerate(step["separated"]):
            if separated and step["pre_sign"][strip] != step["pinned_sign"][strip]:
                return step_index, strip
    return None


def _assert_all_separated_strips_on_attached_sign(capture: dict[str, object]) -> None:
    separated_total = sum(sum(1 for s in step["separated"] if s) for step in capture["steps"])
    assert separated_total > 0, "no strip ever separated, so this test proves nothing"
    mismatch = _first_mismatch(capture)
    assert mismatch is None, (
        f"pinned leading-edge row has the OPPOSITE sign to the attached solve "
        f"at (step {mismatch[0]}, strip {mismatch[1]})"
    )


@pytest.mark.parametrize("arm", ["signed", "signed_mag"])
def test_corrected_pins_put_the_le_row_on_the_attached_sign(arm):
    capture = _run_and_capture({"FLUXV_LE_PIN": arm, "FLUXV_JOINT_SEPARATION": "0"})
    _assert_all_separated_strips_on_attached_sign(capture)


def test_joint_path_puts_the_le_row_on_the_attached_sign():
    capture = _run_and_capture({"FLUXV_LE_PIN": "legacy", "FLUXV_JOINT_SEPARATION": "1"})
    _assert_all_separated_strips_on_attached_sign(capture)


def test_legacy_pin_reproduces_the_sign_defect():
    """The control that makes the three tests above meaningful.

    If this ever PASSES, the legacy behaviour has changed (or the capture broke) and the
    other arms no longer demonstrate a fix.
    """

    capture = _run_and_capture({"FLUXV_LE_PIN": "legacy", "FLUXV_JOINT_SEPARATION": "0"})
    mismatch = _first_mismatch(capture)
    assert mismatch is not None, (
        "the legacy pin no longer produces a sign-opposed leading-edge row"
    )
    step = capture["steps"][mismatch[0]]
    assert step["pre_abs_min"] > 1.0e-6, (
        "attached gamma_pre is near zero, so the sign comparison is not meaningful"
    )


def test_default_is_legacy_and_invalid_values_are_rejected():
    from fluxvortex.warp_fsi import q16_flux_v5m_native as module

    assert module._LE_PIN == os.environ.get("FLUXV_LE_PIN", "legacy")
    environment = dict(os.environ)
    environment["FLUXV_LE_PIN"] = "sideways"
    environment["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "platform"),
         os.path.join(REPO_ROOT, "platform", "warp_vpm"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import fluxvortex.warp_fsi.q16_flux_v5m_native"],
        capture_output=True, text=True, env=environment, cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "FLUXV_LE_PIN" in completed.stderr
