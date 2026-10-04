from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from afp_operations.policy import DatedPolicyParams

_CENTS = Decimal("0.01")
_SIX = Decimal("0.000001")


def _finite_decimal(value: Decimal, *, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def money(value: Decimal) -> Decimal:
    return _finite_decimal(value, name="money").quantize(_CENTS, rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class ContributionBreakdown:
    taxable_wage: Decimal
    worker: Decimal
    admin_fee: Decimal
    insurance_fee: Decimal
    net_to_account: Decimal
    policy_version: str

    def model_dump_jsonable(self) -> dict[str, str]:
        return {
            "taxable_wage": str(self.taxable_wage),
            "worker": str(self.worker),
            "admin_fee": str(self.admin_fee),
            "insurance_fee": str(self.insurance_fee),
            "net_to_account": str(self.net_to_account),
            "policy_version": self.policy_version,
        }


@dataclass(slots=True)
class Account:
    balance: Decimal = Decimal("0.00")

    def __post_init__(self) -> None:
        self.balance = money(self.balance)
        if self.balance < 0:
            raise ValueError("balance cannot be negative")

    def post_contribution(self, amount: Decimal) -> None:
        amount = money(amount)
        if amount < 0:
            raise ValueError("contribution cannot be negative")
        self.balance = money(self.balance + amount)

    def apply_return(self, rate: Decimal) -> None:
        rate = _finite_decimal(rate, name="rate")
        if rate <= Decimal("-1"):
            raise ValueError("rate must be greater than -1")
        self.balance = money(self.balance * (Decimal("1") + rate))

    def apply_asset_fee(self, fee_rate: Decimal) -> None:
        fee_rate = _finite_decimal(fee_rate, name="fee_rate")
        if fee_rate < 0 or fee_rate >= 1:
            raise ValueError("fee_rate must be in [0, 1)")
        self.balance = money(self.balance * (Decimal("1") - fee_rate))

    def model_dump_jsonable(self) -> dict[str, str]:
        return {"balance": str(self.balance)}


def calculate_contribution(
    taxable_wage: Decimal, policy: DatedPolicyParams
) -> ContributionBreakdown:
    wage = money(_finite_decimal(taxable_wage, name="taxable_wage"))
    if wage < 0:
        raise ValueError("taxable_wage cannot be negative")
    worker = money(wage * policy.contribution_rate)
    admin_fee = money(wage * policy.admin_fee_rate)
    insurance_fee = money(wage * policy.insurance_fee_rate)
    net = money(worker - admin_fee - insurance_fee)
    if net < 0:
        raise ValueError("policy fees exceed worker contribution")
    return ContributionBreakdown(wage, worker, admin_fee, insurance_fee, net, policy.version)


def annuity_factor(survival: Iterable[Decimal], discount_rate: Decimal) -> Decimal:
    try:
        rate = _finite_decimal(discount_rate, name="discount_rate")
        if rate <= Decimal("-1"):
            raise ValueError("discount_rate must be greater than -1")
        values = [_finite_decimal(x, name="survival") for x in survival]
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("invalid annuity input") from exc
    if not values:
        raise ValueError("survival cannot be empty")
    previous = Decimal("1")
    total = Decimal("0")
    for k, prob in enumerate(values):
        if prob < 0 or prob > previous:
            raise ValueError(
                "survival probabilities must be finite, non-negative, and non-increasing"
            )
        total += prob / ((Decimal("1") + rate) ** k)
        previous = prob
    return total.quantize(_SIX, rounding=ROUND_HALF_UP)
