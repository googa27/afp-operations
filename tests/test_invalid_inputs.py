from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

from afp_operations.finance import Account, annuity_factor, calculate_contribution
from afp_operations.ops import ContributionEvent, InMemoryLedger, PensionQuoteRequest
from afp_operations.policy import DatedPolicyParams, PolicyError, get_policy
from afp_operations.transport import AgeGrid, finite_volume_step


def test_invalid_nonfinite_financial_inputs_fail_closed() -> None:
    policy = get_policy("research.cl", date(2026, 1, 1))
    with pytest.raises(ValueError):
        calculate_contribution(Decimal("Infinity"), policy)
    with pytest.raises(ValueError):
        Account(Decimal("-0.01"))
    with pytest.raises(ValueError):
        annuity_factor([Decimal("1"), Decimal("NaN")], Decimal("0.02"))


def test_invalid_policy_and_transport_inputs_fail_closed() -> None:
    with pytest.raises(PolicyError):
        get_policy("research.cl", date(2035, 1, 1))
    with pytest.raises(PolicyError):
        get_policy("unsupported", date(2026, 1, 1))
    with pytest.raises(ValueError):
        finite_volume_step(np.array([1.0, np.nan]), AgeGrid(0, 2, 1), 1.0, 0.0)
    with pytest.raises(ValueError):
        finite_volume_step(np.array([1.0, 2.0]), AgeGrid(0, 2, 1), 2.0, 0.0)


def test_policy_account_date_age_and_currency_validations_fail_closed() -> None:
    with pytest.raises(ValueError):
        DatedPolicyParams(
            jurisdiction="research.cl",
            version="bad",
            valid_from=date(2030, 1, 1),
            valid_to=date(2020, 1, 1),
            contribution_rate=Decimal("0.10"),
            admin_fee_rate=Decimal("0.01"),
            insurance_fee_rate=Decimal("0.01"),
            asset_fee_rate=Decimal("0.01"),
            retirement_age=-1,
            official=False,
            disclaimer="bad fixture",
        )
    ledger = InMemoryLedger(currency="CLP")
    with pytest.raises(ValueError):
        ledger.record_contribution(
            ContributionEvent(
                "m", datetime(2026, 1, 1, 1, 2), Decimal("1"), "research.cl", currency="CLP"
            )
        )
    with pytest.raises(ValueError):
        ledger.record_contribution(
            ContributionEvent("m", date(2026, 1, 1), Decimal("1"), "research.cl", currency="USD")
        )
    with pytest.raises(ValueError):
        ledger.quote_pension(
            PensionQuoteRequest(
                member_id="", survival=[Decimal("1")], discount_rate=Decimal("0.01")
            )
        )
