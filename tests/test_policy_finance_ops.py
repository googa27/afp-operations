from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from afp_operations.finance import Account, annuity_factor, calculate_contribution
from afp_operations.ops import ContributionEvent, InMemoryLedger, PensionQuoteRequest
from afp_operations.policy import PolicyError, get_policy


def test_policy_is_explicit_versioned_and_date_checked() -> None:
    policy = get_policy("research.cl", date(2026, 1, 1), version="research-2026-10")
    assert policy.jurisdiction == "research.cl"
    assert policy.official is False
    assert "not official" in policy.disclaimer.lower()
    assert policy.contribution_rate == Decimal("0.10")
    with pytest.raises(PolicyError):
        get_policy("CL", date(2026, 1, 1))
    with pytest.raises(PolicyError):
        get_policy("research.cl", date(1999, 1, 1))


def test_contribution_decimal_account_and_annuity_are_finite() -> None:
    policy = get_policy("research.cl", date(2026, 1, 1))
    result = calculate_contribution(Decimal("1000.00"), policy)
    assert result.worker == Decimal("100.00")
    assert result.admin_fee == Decimal("12.00")
    assert result.net_to_account == Decimal("86.00")
    account = Account(balance=Decimal("100.00"))
    account.apply_return(Decimal("0.05"))
    account.post_contribution(result.net_to_account)
    account.apply_asset_fee(policy.asset_fee_rate)
    assert account.balance == Decimal("190.52")
    af = annuity_factor([Decimal("1"), Decimal("0.9"), Decimal("0.8")], Decimal("0.02"))
    assert af == Decimal("2.651288")
    with pytest.raises(ValueError):
        calculate_contribution(Decimal("NaN"), policy)
    with pytest.raises(ValueError):
        annuity_factor([Decimal("1"), Decimal("1.1")], Decimal("0.02"))


def test_bounded_in_memory_workflow_invariants_no_transfers() -> None:
    ledger = InMemoryLedger()
    event = ContributionEvent(
        member_id="anon-1",
        paid_at=date(2026, 3, 31),
        taxable_wage=Decimal("1000.00"),
        jurisdiction="research.cl",
    )
    posting = ledger.record_contribution(event)
    assert posting.live_transfer is False
    assert posting.net_to_account == Decimal("86.00")
    recon = ledger.reconcile("anon-1")
    assert recon.posted_net == ledger.account("anon-1").balance
    quote = ledger.quote_pension(
        PensionQuoteRequest(
            member_id="anon-1",
            survival=[Decimal("1"), Decimal("0.95")],
            discount_rate=Decimal("0.01"),
        )
    )
    assert quote.annual_pension > Decimal("0")
    with pytest.raises(ValueError):
        ledger.record_contribution(
            ContributionEvent("", date(2026, 1, 1), Decimal("1"), "research.cl")
        )
