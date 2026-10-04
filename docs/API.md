# afp-operations API

Typed research library for a bounded AFP/pension operations vertical slice. It is **not**
real AFP administration, a banking system, legal advice, or official current-law rules.

## Import API

```python
from afp_operations import demo, capabilities
from afp_operations.policy import get_policy
from afp_operations.finance import Account, calculate_contribution, annuity_factor
from afp_operations.ops import InMemoryLedger, ContributionEvent, PensionQuoteRequest
from afp_operations.transport import AgeGrid, finite_volume_step, fem_transport_step
```

- `demo() -> dict[str, Any]`: deterministic finite JSON lifecycle example.
- `capabilities() -> dict[str, dict[str, Any]]`: honest feature status, including
  planned-unavailable live transfers and official legal rules.
- `get_policy("research.cl", date(...), version="research-2026-10")`: explicit
  dated, versioned research fixture only.
- `calculate_contribution`: Decimal contribution split using policy fixture rates.
- `Account`: Decimal balance, returns, contributions, and asset fees.
- `annuity_factor`: present-value factor from non-increasing survival probabilities.
- `InMemoryLedger`: bounded contribution/posting/reconciliation/pension quote workflow;
  no network calls and `live_transfer=False` by construction.
- `finite_volume_step`: conservative age transport with mortality, age-0 inflow,
  open max-age outflow, and mass-balance residual in physical mass units
  (`sum(density) * grid.step`). Mortality uses exponential survival splitting.
- `fem_transport_step`: genuine scikit-fem P1 SUPG implicit transport solve for
  one scalar age-density step. It reports the actual physical mass residual and
  does not substitute a characteristics oracle or clip negative values.
- `afp_operations.bayes.smoke_inference`: optional NumPyro small multigroup MCMC
  smoke when installed with `[bayes]`; country/year/group indexed hierarchical
  log-Poisson mortality with country partial pooling and a random-walk time effect.
- `afp_operations.bayes.posterior_mortality_fv_projection`: propagates posterior
  mortality draws through `finite_volume_step`; contract smoke only, not accepted
  calibration.

## CLI

```bash
afp-operations capabilities
afp-operations demo
```

Both commands emit JSON with `allow_nan=False` compatible finite values.
