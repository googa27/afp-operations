from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from afp_operations import capabilities, demo
from afp_operations.bayes import posterior_mortality_fv_projection, smoke_inference
from afp_operations.transport import AgeGrid


def assert_finite_jsonable(value: object) -> None:
    text = json.dumps(value, allow_nan=False)
    decoded = json.loads(text)
    if isinstance(decoded, dict):
        for item in decoded.values():
            assert_finite_jsonable(item)
    elif isinstance(decoded, list):
        for item in decoded:
            assert_finite_jsonable(item)
    elif isinstance(decoded, float):
        assert np.isfinite(decoded)


def test_root_demo_and_capabilities_are_finite_json() -> None:
    d = demo()
    c = capabilities()
    assert d["research_only"] is True
    assert c["live_transfers"]["status"] == "planned_unavailable"
    assert c["fem_transport"]["status"] == "available"
    assert_finite_jsonable(d)
    assert_finite_jsonable(c)


def test_cli_capabilities_and_demo() -> None:
    for command in ("capabilities", "demo"):
        out = subprocess.check_output(
            [sys.executable, "-m", "afp_operations.cli", command], text=True
        )
        payload = json.loads(out)
        assert isinstance(payload, dict)
        assert_finite_jsonable(payload)


def test_bayes_smoke_is_honest_and_jsonable() -> None:
    result = smoke_inference(num_warmup=5, num_samples=5, seed=1)
    assert result["status"] in {"available", "planned_unavailable"}
    if result["status"] == "available":
        assert result["draws"] == 5
        assert result["divergences"] >= 0
        assert result["observations"] == 12
        assert result["countries"] == 2
        assert result["years"] == 3
        assert result["groups"] == 2
        assert result["posterior_projection"]["status"] == "research_projection_not_calibrated"
        assert result["posterior_projection"]["draws"] == result["draws"]
        assert result["posterior_projection"]["mass_residual_max_abs"] < 1e-10
        assert result["calibration"] == "research smoke only; not accepted calibration"
    assert_finite_jsonable(result)


def test_posterior_mortality_samples_project_to_fv_dynamics_contract() -> None:
    grid = AgeGrid(60.0, 62.0, 1.0)
    samples = {
        "alpha": np.array([-5.0, -4.8]),
        "country_effect": np.array([[0.0], [0.1]]),
        "group_effect": np.array([[0.0], [0.0]]),
        "rw": np.array([[0.0, 0.0], [0.0, 0.2]]),
    }
    projection = posterior_mortality_fv_projection(
        samples,
        density=np.array([100.0, 80.0]),
        grid=grid,
        dt=1.0,
        inflow=0.0,
        country_index=0,
        year_index=1,
        group_index=0,
    )
    assert projection["status"] == "research_projection_not_calibrated"
    assert projection["draws"] == 2
    assert projection["mortality_mean"] > 0.0
    assert len(projection["density_mean"]) == grid.n_cells
    assert projection["density_draws"][0] != projection["density_draws"][1]
    assert_finite_jsonable(projection)
