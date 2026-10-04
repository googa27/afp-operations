# AFP Operations — agent/contributor contract

Research-alpha typed library: explicit dated policy fixtures, Decimal calculations, auditable in-memory lifecycle, McKendrick–von Foerster FV/P1-SUPG FEM, hierarchical dynamic Bayesian mortality. No live transfers, custody, legal determinations, or hidden network calls.

## Canonical docs
- `docs/ARCHITECTURE.yaml`: machine-readable architecture and enforceable limits.
- `docs/API.md`, `docs/THEORY.md`: semantics, units, numerical residuals and caveats.
- `docs/REGULATION.md`: official dated sources, not official current-law implementation.
- `docs/ECOSYSTEM.md`, `docs/ROADMAP.md`: public contracts and tracked extensions.

## Setup and gates
```bash
uv sync --all-extras
uv run --no-sync afp-operations capabilities
uv run --no-sync afp-operations demo
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv build
uv run --no-sync python scripts/publication_guard.py --mode git
python scripts/scan_publication.py
```
Optional Bayesian tests are genuine short synthetic MCMC smoke, not calibrated inference. Limit JAX to CPU when needed.

## Boundaries and extensions
- `policy` owns versioned research parameters; `finance` owns lawful arithmetic; `ops` composes lifecycle; `transport` owns numerical kernels; `bayes` imports the optional inference stack lazily; `cli` exposes the public API.
- Keep public interfaces typed. Use named methods for policy-sensitive operations; no ambiguous arithmetic dunders.
- Add a unit test before a correction. Add integration/e2e tests for new public workflows and architecture fitness tests for new boundaries.
- Maximum 10 runtime Python entries/package directories per source directory, excluding `__init__.py` and metadata; modules <=500 physical lines. Exceptions require scoped reason, owner, risk and refactoring trigger.
- Add country support through explicit dated provider contracts, never silently inferred legal defaults.
- No private member data, credentials, raw portal exports, agent histories or internal paths. Public sources require lineage, vintage, units, grain, quality and license checks. Secrets stay outside this repo.
- Definition of done: real tests/lint/types/build/privacy checks pass; README, contracts, architecture and roadmap agree. Production/compliance claims require evidence beyond research-alpha tests.
