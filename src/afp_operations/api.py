"""Capability discovery and reproducible public synthetic workflows."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np

from afp_operations.ops import ContributionEvent, InMemoryLedger, PensionQuoteRequest
from afp_operations.transport import AgeGrid, fem_transport_step, finite_volume_step


def capabilities() -> dict[str, dict[str, Any]]:
    """Return honest machine-readable feature capability status."""
    return {
        "policy_fixtures": {
            "status": "available",
            "scope": "explicit dated research parameters; not official current law",
        },
        "decimal_accounts": {
            "status": "available",
            "scope": "bounded in-memory posting, fees, reconciliation, quotes",
        },
        "finite_volume_transport": {
            "status": "available",
            "scope": "conservative age transport with mortality, inflow, outflow, physical mass balance",
        },
        "fem_transport": {
            "status": "available",
            "scope": "genuine scikit-fem P1 SUPG implicit solve; physical mass residuals reported; no output clipping",
        },
        "bayes_numpyro": {
            "status": "optional",
            "scope": "country/year/group log-Poisson hierarchy; actual posterior-to-FV synthetic smoke",
            "cli": "afp-operations bayes-smoke",
        },
        "live_transfers": {
            "status": "planned_unavailable",
            "scope": "intentionally absent; no banking, custody, or live AFP transfers",
        },
        "official_legal_rules": {
            "status": "planned_unavailable",
            "scope": "requires jurisdiction-specific legal validation outside this alpha",
        },
    }


def demo() -> dict[str, Any]:
    """Run an actual policy/account/reconciliation/quote/FV/FEM synthetic lifecycle."""
    ledger = InMemoryLedger()
    event = ContributionEvent(
        member_id="demo-member",
        paid_at=date(2026, 1, 31),
        taxable_wage=Decimal("1000.00"),
        jurisdiction="research.cl",
    )
    posting = ledger.record_contribution(event)
    quote = ledger.quote_pension(
        PensionQuoteRequest(
            member_id="demo-member",
            survival=[Decimal("1.0"), Decimal("0.96"), Decimal("0.91")],
            discount_rate=Decimal("0.02"),
        )
    )
    grid = AgeGrid(0.0, 4.0, 1.0)
    density, report = finite_volume_step(np.array([10.0, 9.0, 8.0, 7.0]), grid, 1.0, 0.02, 5.0)
    fem = fem_transport_step([10.0, 9.0, 8.0, 7.0], grid, 1.0, 0.02, 5.0)
    return {
        "research_only": True,
        "policy": ledger.policies["demo-member"].model_dump_jsonable(),
        "posting": posting.model_dump_jsonable(),
        "reconciliation": ledger.reconcile("demo-member").model_dump_jsonable(),
        "quote": quote.model_dump_jsonable(),
        "transport": {"density": density.tolist(), "mass_report": report.model_dump_jsonable()},
        "transport_fem": fem.model_dump_jsonable(),
    }
