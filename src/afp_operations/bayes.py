from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from afp_operations.transport import AgeGrid, FloatArray, finite_volume_step


def mortality_hierarchy_model(
    deaths: Any,
    exposure: Any,
    country: Any,
    year: Any,
    group: Any,
    n_years: int,
    n_countries: int,
    n_groups: int,
) -> None:
    """Hierarchical country/year/group log-Poisson mortality model.

    Optional NumPyro model with lazy imports. Observation arrays may have any length;
    ``country``, ``year``, and ``group`` are explicit observation-level indices, so the
    number of calendar years is not assumed to equal the number of observations.
    """
    import jax.numpy as jnp  # type: ignore[import-not-found]
    import numpyro  # type: ignore[import-not-found,import-untyped]
    import numpyro.distributions as dist  # type: ignore[import-not-found,import-untyped]

    alpha = numpyro.sample("alpha", dist.Normal(-5.0, 1.0))
    sigma_country = numpyro.sample("sigma_country", dist.HalfNormal(0.4))
    sigma_group = numpyro.sample("sigma_group", dist.HalfNormal(0.4))
    sigma_rw = numpyro.sample("sigma_rw", dist.HalfNormal(0.15))
    with numpyro.plate("country", n_countries):
        country_raw = numpyro.sample("country_raw", dist.Normal(0.0, 1.0))
    with numpyro.plate("group", n_groups):
        group_raw = numpyro.sample("group_raw", dist.Normal(0.0, 1.0))
    with numpyro.plate("rw_steps", n_years):
        rw_eps = numpyro.sample("rw_eps", dist.Normal(0.0, 1.0))
    country_effect = numpyro.deterministic("country_effect", sigma_country * country_raw)
    group_effect = numpyro.deterministic("group_effect", sigma_group * group_raw)
    rw = numpyro.deterministic("rw", jnp.cumsum(rw_eps * sigma_rw))
    log_rate = alpha + country_effect[country] + group_effect[group] + rw[year]
    with numpyro.plate("obs", deaths.shape[0]):
        numpyro.sample("deaths", dist.Poisson(exposure * jnp.exp(log_rate)), obs=deaths)


def _synthetic_multigroup_counts() -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    int,
    int,
    int,
]:
    countries = 2
    years = 3
    groups = 2
    rows: list[tuple[int, int, int]] = []
    for country in range(countries):
        for year in range(years):
            for group in range(groups):
                rows.append((country, year, group))
    idx_country = np.array([r[0] for r in rows], dtype=np.int32)
    idx_year = np.array([r[1] for r in rows], dtype=np.int32)
    idx_group = np.array([r[2] for r in rows], dtype=np.int32)
    exposure = np.full(len(rows), 1_000.0, dtype=np.float32)
    # Deterministic small counts with country/year/group variation; synthetic only.
    base_rate = np.exp(-5.2 + 0.25 * idx_country + 0.12 * idx_year + 0.18 * idx_group)
    deaths = np.maximum(1, np.rint(exposure * base_rate)).astype(np.float32)
    return deaths, exposure, idx_country, idx_year, idx_group, countries, years, groups


def smoke_inference(num_warmup: int = 10, num_samples: int = 10, seed: int = 0) -> dict[str, Any]:
    """Run a tiny real multigroup NumPyro MCMC smoke when optional dependencies exist.

    Returned diagnostics are intentionally labelled as smoke diagnostics, not accepted
    calibration for actuarial or policy claims.
    """
    try:
        import jax.numpy as jnp  # type: ignore[import-not-found]
        import numpyro  # type: ignore[import-not-found,import-untyped]
        from jax import random  # type: ignore[import-not-found]
        from numpyro.infer import MCMC, NUTS  # type: ignore[import-not-found,import-untyped]
    except Exception as exc:
        return {
            "status": "planned_unavailable",
            "reason": f"Install afp-operations[bayes] for NumPyro smoke inference ({type(exc).__name__}).",
        }
    if num_warmup <= 0 or num_samples <= 0:
        raise ValueError("num_warmup and num_samples must be positive")
    deaths_np, exposure_np, country_np, year_np, group_np, countries, years, groups = (
        _synthetic_multigroup_counts()
    )
    deaths = jnp.asarray(deaths_np)
    exposure = jnp.asarray(exposure_np)
    country = jnp.asarray(country_np)
    year = jnp.asarray(year_np)
    group = jnp.asarray(group_np)
    numpyro.set_host_device_count(1)
    kernel = NUTS(mortality_hierarchy_model, target_accept_prob=0.75)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False
    )
    mcmc.run(
        random.PRNGKey(seed),
        deaths=deaths,
        exposure=exposure,
        country=country,
        year=year,
        group=group,
        n_years=years,
        n_countries=countries,
        n_groups=groups,
    )
    samples = mcmc.get_samples()
    extras = mcmc.get_extra_fields()
    alpha_mean = float(np.asarray(samples["alpha"]).mean())
    divergences = int(np.asarray(extras.get("diverging", np.array([], dtype=bool))).sum())
    return {
        "status": "available",
        "draws": int(num_samples),
        "chains": 1,
        "observations": int(deaths_np.size),
        "countries": countries,
        "years": years,
        "groups": groups,
        "alpha_mean": alpha_mean,
        "divergences": divergences,
        "diagnostics": "short-run multigroup smoke only; run longer chains and ArviZ R-hat/ESS before inference use",
        "posterior_projection": posterior_mortality_fv_projection(
            samples,
            density=np.array([100.0, 80.0]),
            grid=AgeGrid(60.0, 62.0, 1.0),
            dt=1.0,
            inflow=0.0,
            country_index=0,
            year_index=2,
            group_index=0,
        ),
        "calibration": "research smoke only; not accepted calibration",
    }


def _draw_vector(samples: dict[str, Any], name: str, draws: int) -> FloatArray:
    value = np.asarray(samples[name], dtype=float)
    if value.shape[0] != draws:
        raise ValueError(f"sample {name!r} must have leading draw dimension {draws}")
    return value


def posterior_mortality_fv_projection(
    samples: dict[str, Any],
    *,
    density: FloatArray,
    grid: AgeGrid,
    dt: float,
    inflow: float,
    country_index: int,
    year_index: int,
    group_index: int,
) -> dict[str, Any]:
    """Project posterior mortality draws through the finite-volume transport step.

    This is a contract/smoke utility: it propagates sampled log mortality rates into
    deterministic FV dynamics and labels the result as not calibrated.
    """
    alpha = np.asarray(samples["alpha"], dtype=float)
    if alpha.ndim != 1 or alpha.size == 0 or not np.all(np.isfinite(alpha)):
        raise ValueError("alpha samples must be a non-empty finite vector")
    draws = alpha.shape[0]
    country_effect = _draw_vector(samples, "country_effect", draws)
    group_effect = _draw_vector(samples, "group_effect", draws)
    rw = _draw_vector(samples, "rw", draws)
    if country_index < 0 or group_index < 0 or year_index < 0:
        raise ValueError("country/year/group indices must be non-negative")
    if (
        country_index >= country_effect.shape[1]
        or group_index >= group_effect.shape[1]
        or year_index >= rw.shape[1]
    ):
        raise ValueError("country/year/group index out of posterior sample bounds")
    log_mu = (
        alpha + country_effect[:, country_index] + group_effect[:, group_index] + rw[:, year_index]
    )
    mortality = np.exp(log_mu)
    density_draws: list[list[float]] = []
    residuals: list[float] = []
    for mu in mortality:
        projected, report = finite_volume_step(
            density, grid, dt=dt, mortality=float(mu), inflow=inflow
        )
        density_draws.append(projected.tolist())
        residuals.append(report.residual)
    density_array = np.asarray(density_draws, dtype=float)
    return {
        "status": "research_projection_not_calibrated",
        "draws": int(draws),
        "mortality_mean": float(np.mean(mortality)),
        "mortality_min": float(np.min(mortality)),
        "mortality_max": float(np.max(mortality)),
        "density_mean": np.mean(density_array, axis=0).tolist(),
        "density_draws": density_draws,
        "mass_residual_max_abs": float(np.max(np.abs(residuals))),
        "calibration": "posterior-to-FV contract only; not accepted actuarial calibration",
    }
