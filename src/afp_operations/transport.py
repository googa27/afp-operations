from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AgeGrid:
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        if not all(np.isfinite([self.start, self.stop, self.step])):
            raise ValueError("age grid values must be finite")
        if self.start < 0:
            raise ValueError("age grid start cannot be negative")
        if self.step <= 0 or self.stop <= self.start:
            raise ValueError("age grid requires stop > start and step > 0")
        cells = (self.stop - self.start) / self.step
        if abs(cells - round(cells)) > 1e-10:
            raise ValueError("age range must be an integer number of cells")

    @property
    def n_cells(self) -> int:
        return int(round((self.stop - self.start) / self.step))

    @property
    def cell_centers(self) -> FloatArray:
        return (self.start + (np.arange(self.n_cells, dtype=float) + 0.5) * self.step).astype(float)

    @property
    def nodes(self) -> FloatArray:
        return np.linspace(self.start, self.stop, self.n_cells + 1, dtype=float)


@dataclass(frozen=True, slots=True)
class MassBalanceReport:
    initial_mass: float
    final_mass: float
    births: float
    deaths: float
    outflow: float
    residual: float

    def model_dump_jsonable(self) -> dict[str, float]:
        return {
            "initial_mass": self.initial_mass,
            "final_mass": self.final_mass,
            "births": self.births,
            "deaths": self.deaths,
            "outflow": self.outflow,
            "residual": self.residual,
        }


@dataclass(frozen=True, slots=True)
class FemResult:
    density: FloatArray
    mass_report: MassBalanceReport
    method: str
    stabilization: str
    assembly: dict[str, float]

    def model_dump_jsonable(self) -> dict[str, object]:
        return {
            "density": self.density.tolist(),
            "mass_report": self.mass_report.model_dump_jsonable(),
            "method": self.method,
            "stabilization": self.stabilization,
            "assembly": self.assembly,
        }


def _as_density(density: NDArray[np.floating[Any]] | list[float], grid: AgeGrid) -> FloatArray:
    arr = np.asarray(density, dtype=float)
    if arr.shape != (grid.n_cells,):
        raise ValueError(f"density shape must be ({grid.n_cells},)")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("density must be finite and non-negative")
    return arr.astype(float, copy=True)


def _mortality_array(mortality: float | NDArray[np.floating[Any]], n: int) -> FloatArray:
    mu = np.asarray(mortality, dtype=float)
    if mu.shape == ():
        mu = np.full(n, float(mu), dtype=float)
    if mu.shape != (n,) or not np.all(np.isfinite(mu)) or np.any(mu < 0):
        raise ValueError("mortality must be finite, non-negative scalar or cell array")
    return mu.astype(float, copy=False)


def finite_volume_step(
    density: NDArray[np.floating[Any]] | list[float],
    grid: AgeGrid,
    dt: float,
    mortality: float | NDArray[np.floating[Any]],
    inflow: float = 0.0,
) -> tuple[FloatArray, MassBalanceReport]:
    """Conservative upwind age-transport step with exact mortality splitting.

    ``density`` is a per-age density over cells. Reported masses are physical people
    counts, i.e. cell-density sums multiplied by ``grid.step``. Mortality is applied
    by exponential survival ``exp(-mu * dt)`` rather than by clipping an explicit Euler
    hazard step.
    """
    arr = _as_density(density, grid)
    if not np.isfinite(dt) or dt <= 0 or dt > grid.step:
        raise ValueError("dt must be finite and in (0, grid.step]")
    if not np.isfinite(inflow) or inflow < 0:
        raise ValueError("inflow must be finite and non-negative")
    cfl = dt / grid.step
    mu = _mortality_array(mortality, grid.n_cells)
    survived = arr * np.exp(-mu * dt)
    deaths = float(np.sum((arr - survived) * grid.step))
    nxt = np.empty_like(arr)
    nxt[0] = (1.0 - cfl) * survived[0] + cfl * inflow
    if grid.n_cells > 1:
        nxt[1:] = (1.0 - cfl) * survived[1:] + cfl * survived[:-1]
    outflow = float(dt * survived[-1])
    births = float(dt * inflow)
    initial = float(np.sum(arr) * grid.step)
    final = float(np.sum(nxt) * grid.step)
    residual = final - initial - births + deaths + outflow
    return nxt, MassBalanceReport(initial, final, births, deaths, outflow, float(residual))


def constant_hazard_characteristics(
    density: NDArray[np.floating[Any]] | list[float],
    dt: float,
    da: float,
    mortality: float,
    inflow: float = 0.0,
) -> FloatArray:
    """Exact constant-hazard characteristics oracle for the exact cell-shift gate."""
    arr = np.asarray(density, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("density must be finite non-negative vector")
    if (
        not all(np.isfinite([dt, da, mortality, inflow]))
        or dt <= 0
        or da <= 0
        or mortality < 0
        or inflow < 0
    ):
        raise ValueError("dt, da, mortality, and inflow must be valid finite values")
    cfl = dt / da
    if abs(cfl - 1.0) > 1e-12:
        raise ValueError("characteristics oracle currently supports exact cell shift dt == da")
    out = np.empty_like(arr, dtype=float)
    out[0] = inflow
    if arr.size > 1:
        out[1:] = arr[:-1] * np.exp(-mortality * dt)
    return out


def _cell_averages_from_nodes(nodes_values: FloatArray) -> FloatArray:
    return (0.5 * (nodes_values[:-1] + nodes_values[1:])).astype(float)


def _nodes_from_cell_averages(cell_values: FloatArray) -> FloatArray:
    nodes = np.empty(cell_values.size + 1, dtype=float)
    nodes[0] = cell_values[0]
    nodes[-1] = cell_values[-1]
    if cell_values.size > 1:
        nodes[1:-1] = 0.5 * (cell_values[:-1] + cell_values[1:])
    return nodes


def _fem_mass_report(
    initial_density: FloatArray,
    final_density: FloatArray,
    final_nodes: FloatArray,
    grid: AgeGrid,
    dt: float,
    mortality: float,
    inflow: float,
) -> MassBalanceReport:
    initial = float(np.sum(initial_density) * grid.step)
    final = float(np.sum(final_density) * grid.step)
    births = float(dt * inflow)
    deaths = float(dt * mortality * np.sum(final_density) * grid.step)
    outflow = float(dt * final_nodes[-1])
    residual = final - initial - births + deaths + outflow
    return MassBalanceReport(initial, final, births, deaths, outflow, float(residual))


def fem_transport_step(
    density: NDArray[np.floating[Any]] | list[float],
    grid: AgeGrid,
    dt: float,
    mortality: float,
    inflow: float = 0.0,
) -> FemResult:
    """Solve one P1 SUPG implicit FEM step for age transport.

    The solved weak form is backward Euler for ``n_t + n_a + mu n = 0`` with an
    essential inflow value at age ``grid.start`` and natural outflow at ``grid.stop``:

    ``(u, v + tau v_a) + dt (u_a + mu u, v + tau v_a) = (u_old, v + tau v_a)``.

    The implementation assembles the P1 matrices with scikit-fem and solves the
    condensed linear system with SciPy. It returns the solved cell averages directly;
    no characteristic oracle substitution and no positivity clipping are applied.
    """
    arr = _as_density(density, grid)
    if not all(np.isfinite([dt, mortality, inflow])) or dt <= 0 or dt > grid.step:
        raise ValueError("FEM step requires finite dt in (0, grid.step]")
    if mortality < 0 or inflow < 0:
        raise ValueError("mortality and inflow must be non-negative")
    try:
        from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]
        from skfem import (  # type: ignore[import-untyped]
            Basis,
            BilinearForm,
            ElementLineP1,
            MeshLine,
            asm,
        )
        from skfem.helpers import grad  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - dependency declared in base package
        raise RuntimeError("scikit-fem and SciPy are required for fem_transport_step") from exc

    mesh = MeshLine(grid.nodes)
    basis = Basis(mesh, ElementLineP1())
    tau = grid.step / 2.0

    @BilinearForm  # type: ignore[untyped-decorator]
    def mass(u: Any, v: Any, _w: Any) -> Any:
        return u * v

    @BilinearForm  # type: ignore[untyped-decorator]
    def test_streamline(u: Any, v: Any, _w: Any) -> Any:
        return u * grad(v)[0]

    @BilinearForm  # type: ignore[untyped-decorator]
    def advection(u: Any, v: Any, _w: Any) -> Any:
        return grad(u)[0] * v

    @BilinearForm  # type: ignore[untyped-decorator]
    def streamline_diffusion(u: Any, v: Any, _w: Any) -> Any:
        return grad(u)[0] * grad(v)[0]

    mtx = asm(mass, basis)
    btx = asm(test_streamline, basis)
    ctx = asm(advection, basis)
    ktx = asm(streamline_diffusion, basis)
    lhs = mtx + tau * btx + dt * (ctx + tau * ktx + mortality * mtx + mortality * tau * btx)
    rhs_operator = mtx + tau * btx
    old_nodes = _nodes_from_cell_averages(arr)
    rhs = np.asarray(rhs_operator @ old_nodes, dtype=float)

    # Strongly enforce the inflow Dirichlet degree of freedom by exact elimination.
    n_dofs = old_nodes.size
    known = 0
    unknown = np.arange(1, n_dofs)
    rhs_unknown = rhs[unknown] - np.asarray(lhs[unknown, known].toarray()).reshape(-1) * inflow
    lhs_unknown = lhs[unknown][:, unknown].tocsc()
    solved_unknown = np.asarray(spsolve(lhs_unknown, rhs_unknown), dtype=float)
    solved_nodes = np.empty(n_dofs, dtype=float)
    solved_nodes[known] = inflow
    solved_nodes[unknown] = solved_unknown
    solved_cells = _cell_averages_from_nodes(solved_nodes)
    report = _fem_mass_report(arr, solved_cells, solved_nodes, grid, dt, mortality, inflow)
    return FemResult(
        density=solved_cells,
        mass_report=report,
        method="scikit-fem P1 SUPG implicit transport solve",
        stabilization="SUPG tau=h/2 numerical stabilization; solved density returned with no clipping",
        assembly={
            "mass_nnz": float(mtx.nnz),
            "supg_lhs_nnz": float(lhs.nnz),
            "tau": float(tau),
            "solution_min": float(np.min(solved_cells)),
            "solution_max": float(np.max(solved_cells)),
        },
    )
