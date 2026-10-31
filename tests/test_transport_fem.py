from __future__ import annotations

import numpy as np
import pytest

from afp_operations.transport import (
    AgeGrid,
    constant_hazard_characteristics,
    fem_transport_step,
    finite_volume_step,
)


def test_finite_volume_mass_balance_and_boundary_outflow() -> None:
    grid = AgeGrid(0.0, 5.0, 1.0)
    density = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    next_density, report = finite_volume_step(density, grid, dt=1.0, mortality=0.1, inflow=7.0)
    survival = np.exp(-0.1)
    assert np.all(next_density >= 0.0)
    assert np.allclose(
        next_density, [7.0, 10.0 * survival, 20.0 * survival, 30.0 * survival, 40.0 * survival]
    )
    assert report.outflow == pytest.approx(50.0 * survival)
    assert report.births == 7.0
    assert report.deaths == pytest.approx(np.sum(density * (1.0 - survival)))
    assert report.residual == pytest.approx(0.0)
    with pytest.raises(ValueError):
        finite_volume_step(np.array([1.0, np.inf]), AgeGrid(0, 2, 1), dt=1.0, mortality=0.0)
    with pytest.raises(ValueError):
        AgeGrid(0, 1, 0)


def test_constant_hazard_characteristics_oracle() -> None:
    density = np.array([10.0, 20.0, 30.0, 40.0])
    oracle = constant_hazard_characteristics(density, dt=1.0, da=1.0, mortality=0.2, inflow=5.0)
    assert np.allclose(oracle, [5.0, 10.0 * np.exp(-0.2), 20.0 * np.exp(-0.2), 30.0 * np.exp(-0.2)])
    with pytest.raises(ValueError):
        constant_hazard_characteristics(density, dt=-1.0, da=1.0, mortality=0.0)


def test_fem_adapter_matches_constant_hazard_characteristics_oracle() -> None:
    grid = AgeGrid(0.0, 8.0, 1.0)
    density = np.linspace(1.0, 8.0, grid.n_cells)
    fem = fem_transport_step(density, grid, dt=1.0, mortality=0.05, inflow=3.0)
    oracle = constant_hazard_characteristics(density, dt=1.0, da=1.0, mortality=0.05, inflow=3.0)
    assert fem.method == "scikit-fem P1 SUPG implicit transport solve"
    assert "no clipping" in fem.stabilization
    assert np.all(np.isfinite(fem.density))
    assert np.linalg.norm(fem.density - oracle, ord=np.inf) < 1.0
    assert abs(fem.mass_report.residual) < 0.2


def test_convergence_to_characteristics_under_refinement() -> None:
    errors: list[float] = []
    for da in (1.0, 0.5):
        grid = AgeGrid(0.0, 10.0, da)
        ages = grid.cell_centers
        density = 1.0 + ages / 10.0
        fv, _ = finite_volume_step(density, grid, dt=da, mortality=0.03, inflow=1.25)
        oracle = constant_hazard_characteristics(density, dt=da, da=da, mortality=0.03, inflow=1.25)
        errors.append(float(np.linalg.norm(fv - oracle, ord=np.inf)))
    assert errors[1] <= errors[0] + 1e-12


def test_finite_volume_non_unit_age_cells_physical_mass_and_exponential_survival() -> None:
    grid = AgeGrid(0.0, 2.0, 0.5)
    density = np.array([4.0, 6.0, 8.0, 10.0])
    next_density, report = finite_volume_step(density, grid, dt=0.25, mortality=3.0, inflow=2.0)
    survived = density * np.exp(-3.0 * 0.25)
    expected = np.array(
        [
            0.5 * survived[0] + 0.5 * 2.0,
            0.5 * survived[1] + 0.5 * survived[0],
            0.5 * survived[2] + 0.5 * survived[1],
            0.5 * survived[3] + 0.5 * survived[2],
        ]
    )
    assert np.allclose(next_density, expected)
    assert report.initial_mass == pytest.approx(np.sum(density) * grid.step)
    assert report.final_mass == pytest.approx(np.sum(next_density) * grid.step)
    assert report.births == pytest.approx(0.25 * 2.0)
    assert report.deaths == pytest.approx(np.sum((density - survived) * grid.step))
    assert report.outflow == pytest.approx(0.25 * survived[-1])
    assert report.residual == pytest.approx(0.0)


def test_fem_supg_solve_converges_to_smooth_manufactured_transport_solution() -> None:
    mortality = 0.07
    final_errors: list[float] = []
    for step in (0.1, 0.05):
        grid = AgeGrid(0.0, 1.0, step)
        old_density = 1.0 + 0.2 * np.sin(2.0 * np.pi * grid.cell_centers)
        inflow = float(1.0 + 0.2 * np.sin(-2.0 * np.pi * step))
        fem = fem_transport_step(old_density, grid, dt=step, mortality=mortality, inflow=inflow)
        exact = (1.0 + 0.2 * np.sin(2.0 * np.pi * (grid.cell_centers - step))) * np.exp(
            -mortality * step
        )
        final_errors.append(float(np.sqrt(np.mean((fem.density - exact) ** 2))))
        assert fem.assembly["solution_min"] > 0.0
    assert final_errors[1] < 0.65 * final_errors[0]


def test_fem_preserves_constant_steady_profile_and_physical_mass() -> None:
    grid = AgeGrid(2.0, 4.0, 0.1)
    density = np.full(grid.n_cells, 3.0)
    for _ in range(10):
        result = fem_transport_step(density, grid, dt=0.1, mortality=0.0, inflow=3.0)
        density = result.density
        assert result.mass_report.final_mass == pytest.approx(6.0)
        assert result.mass_report.residual == pytest.approx(0.0, abs=2e-13)
    np.testing.assert_allclose(density, 3.0, atol=2e-13)


def test_fem_converges_over_fixed_time_with_time_dependent_inflow() -> None:
    errors = []
    residuals = []
    for elements in (40, 80, 160):
        grid = AgeGrid(0.0, 10.0, 10.0 / elements)
        ages = grid.cell_centers
        density = 1.0 + 0.2 * np.sin(2.0 * np.pi * ages / 10.0)
        dt = 1.0 / elements
        max_residual = 0.0
        for step in range(elements):
            time = (step + 1) * dt
            inflow = (1.0 + 0.2 * np.sin(-2.0 * np.pi * time / 10.0)) * np.exp(-0.1 * time)
            result = fem_transport_step(density, grid, dt, mortality=0.1, inflow=float(inflow))
            density = result.density
            max_residual = max(max_residual, abs(result.mass_report.residual))
        exact = (1.0 + 0.2 * np.sin(2.0 * np.pi * (ages - 1.0) / 10.0)) * np.exp(-0.1)
        errors.append(float(np.sqrt(np.mean((density - exact) ** 2))))
        residuals.append(max_residual)
    assert errors[1] < 0.65 * errors[0]
    assert errors[2] < 0.65 * errors[1]
    assert residuals[1] < 0.4 * residuals[0]
    assert residuals[2] < 0.4 * residuals[1]
