from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from afp_operations.finance import (
    Account,
    ContributionBreakdown,
    annuity_factor,
    calculate_contribution,
    money,
)
from afp_operations.policy import DatedPolicyParams, get_policy


@dataclass(frozen=True, slots=True)
class ContributionEvent:
    member_id: str
    paid_at: date
    taxable_wage: Decimal
    jurisdiction: str
    policy_version: str | None = None
    currency: str = "CLP"

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("member_id is required")
        if type(self.paid_at) is not date:
            raise ValueError("paid_at must be a date without time-of-day")
        if not self.currency or self.currency != self.currency.upper():
            raise ValueError("currency must be a non-empty uppercase ISO-like code")


@dataclass(frozen=True, slots=True)
class ContributionPosting:
    member_id: str
    paid_at: date
    breakdown: ContributionBreakdown
    live_transfer: bool = False

    @property
    def net_to_account(self) -> Decimal:
        return self.breakdown.net_to_account

    def model_dump_jsonable(self) -> dict[str, str | bool | dict[str, str]]:
        return {
            "member_id": self.member_id,
            "paid_at": self.paid_at.isoformat(),
            "breakdown": self.breakdown.model_dump_jsonable(),
            "net_to_account": str(self.net_to_account),
            "live_transfer": self.live_transfer,
        }


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    member_id: str
    events: int
    posted_net: Decimal
    account_balance: Decimal
    residual: Decimal

    def model_dump_jsonable(self) -> dict[str, str | int]:
        return {
            "member_id": self.member_id,
            "events": self.events,
            "posted_net": str(self.posted_net),
            "account_balance": str(self.account_balance),
            "residual": str(self.residual),
        }


@dataclass(frozen=True, slots=True)
class PensionQuoteRequest:
    member_id: str
    survival: list[Decimal]
    discount_rate: Decimal

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("member_id is required")


@dataclass(frozen=True, slots=True)
class PensionQuote:
    member_id: str
    balance: Decimal
    annuity_factor: Decimal
    annual_pension: Decimal
    disclaimer: str

    def model_dump_jsonable(self) -> dict[str, str]:
        return {
            "member_id": self.member_id,
            "balance": str(self.balance),
            "annuity_factor": str(self.annuity_factor),
            "annual_pension": str(self.annual_pension),
            "disclaimer": self.disclaimer,
        }


class InMemoryLedger:
    """Bounded research workflow; it never moves real money or contacts networks."""

    def __init__(self, *, max_events: int = 10_000, currency: str = "CLP") -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        if not currency or currency != currency.upper():
            raise ValueError("currency must be a non-empty uppercase ISO-like code")
        self.max_events = max_events
        self.currency = currency
        self._accounts: dict[str, Account] = {}
        self._postings: list[ContributionPosting] = []
        self.policies: dict[str, DatedPolicyParams] = {}

    def account(self, member_id: str) -> Account:
        if not member_id:
            raise ValueError("member_id is required")
        return self._accounts.setdefault(member_id, Account())

    def record_contribution(self, event: ContributionEvent) -> ContributionPosting:
        if len(self._postings) >= self.max_events:
            raise ValueError("ledger event bound exceeded")
        if not event.member_id:
            raise ValueError("member_id is required")
        if event.currency != self.currency:
            raise ValueError("contribution currency must match ledger currency")
        policy = get_policy(event.jurisdiction, event.paid_at, event.policy_version)
        breakdown = calculate_contribution(event.taxable_wage, policy)
        posting = ContributionPosting(
            event.member_id, event.paid_at, breakdown, live_transfer=False
        )
        self.account(event.member_id).post_contribution(posting.net_to_account)
        self._postings.append(posting)
        self.policies[event.member_id] = policy
        return posting

    def reconcile(self, member_id: str) -> ReconciliationReport:
        account = self.account(member_id)
        posted = money(
            sum(
                (p.net_to_account for p in self._postings if p.member_id == member_id), Decimal("0")
            )
        )
        residual = money(account.balance - posted)
        events = sum(1 for p in self._postings if p.member_id == member_id)
        return ReconciliationReport(member_id, events, posted, account.balance, residual)

    def quote_pension(self, request: PensionQuoteRequest) -> PensionQuote:
        account = self.account(request.member_id)
        factor = annuity_factor(request.survival, request.discount_rate)
        if factor <= 0:
            raise ValueError("annuity factor must be positive")
        annual = money(account.balance / factor)
        return PensionQuote(
            request.member_id,
            account.balance,
            factor,
            annual,
            "Research quote from synthetic assumptions; not an offer, guarantee, or legal advice.",
        )
