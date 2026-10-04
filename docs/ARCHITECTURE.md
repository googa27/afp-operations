# Architecture

`afp-operations` is a small, public, typed research kernel. The source of truth for automated publication checks is [`docs/ARCHITECTURE.yaml`](ARCHITECTURE.yaml).

## Layers

```text
CLI / public API
  ├─ policy fixtures          explicit dated research assumptions
  ├─ finance/account math     Decimal contribution, fees, balance, quote
  ├─ operations ledger        bounded in-memory posting/reconciliation
  ├─ transport kernels        finite-volume oracle + scikit-fem adapter
  └─ optional bayes smoke     tiny NumPyro diagnostics path when installed
```

## Boundary rules

- No live transfers, custody, banking, or hidden network calls.
- No raw private member data, raw workbooks, PDFs, databases, credentials, or agent histories.
- Policy fixtures are not official Chilean current law.
- Synthetic demo calculations must be labelled synthetic.
- Downstream workbenches should import the package or call the CLI, not reach into private provider details.

## Quality gates

The public CI and local publication checks run tests, Ruff, mypy, build, the publication guard, and redacted Gitleaks. Source files have a 500-line no-growth baseline and source directories have a 10-file default limit unless documented in `docs/ARCHITECTURE.yaml`.
