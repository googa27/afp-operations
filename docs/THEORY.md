# THEORY

This package implements a compact research-first pension operations kernel.

## Policy parameters

The included `research.cl` policy is an explicit dated fixture, version
`research-2026-10`, valid 2020-01-01 to 2030-12-31. It is deliberately not labelled
as official Chilean law. Calculations using it are reproducible tests of software
contracts, not legal or administrative determinations.

## Accounts and annuities

Money-like account operations use `Decimal` and cent rounding. Contributions split a
taxable wage into worker contribution, admin fee, insurance fee, and net account
posting. Annuity factors compute `sum_k survival[k] / (1+r)^k` and require finite,
non-negative, non-increasing survival.

## Age transport

Population mass follows the McKendrick-von Foerster/cohort-component transport idea:
people age at unit speed, die through explicit mortality, enter through age-0 inflow,
and leave through the maximum-age outflow boundary. The finite-volume update is
locally conservative and reports births, deaths, outflow, and residual in physical
mass units: cell densities are per-age, so total mass is `sum(density) * grid.step`.
Mortality is applied by exact exponential survival splitting `exp(-mu dt)`; no large
hazard is silently clipped. There is no artificial age diffusion.

## FEM adapter

The FEM route solves a one-dimensional scikit-fem P1 SUPG weak form for the same
transport residual using backward Euler:

```text
(u, v + tau v_a) + dt (u_a + mu u, v + tau v_a)
  = (u_old, v + tau v_a)
```

with an essential inflow value at age 0 and natural outflow at the maximum age.
The returned density is the solved P1 cell-average projection itself; it is not a
characteristics oracle substitution. SUPG is numerical stabilization, not a demographic
process. Outputs are not clipped, and the mass-balance residual is the reported physical
balance residual rather than a fabricated zero.

## Bayesian smoke

The optional NumPyro model is a country/year/group indexed hierarchical log-Poisson
mortality model for synthetic death/exposure counts. It has partial-pooling country
intercepts, group effects, and a random-walk calendar-time effect. `smoke_inference`
runs only a tiny CPU MCMC to verify optional dependencies and model indexing. The
`posterior_mortality_fv_projection` helper propagates posterior mortality draws through
the finite-volume solver to verify the deterministic coupling contract. These outputs
are research smoke only and are not accepted calibration, actuarial certification, or
policy evidence.
