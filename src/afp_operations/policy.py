from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


def _validate_rate(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0 or value >= 1:
        raise ValueError(f"{name} must be a finite Decimal in [0, 1)")


class PolicyError(ValueError):
    """Raised when no explicit dated research policy fixture applies."""


@dataclass(frozen=True, slots=True)
class DatedPolicyParams:
    jurisdiction: str
    version: str
    valid_from: date
    valid_to: date
    contribution_rate: Decimal
    admin_fee_rate: Decimal
    insurance_fee_rate: Decimal
    asset_fee_rate: Decimal
    retirement_age: int
    official: bool
    disclaimer: str

    def __post_init__(self) -> None:
        if not self.jurisdiction or not self.version:
            raise ValueError("jurisdiction and version are required")
        if type(self.valid_from) is not date or type(self.valid_to) is not date:
            raise ValueError("policy validity bounds must be date instances")
        if self.valid_to < self.valid_from:
            raise ValueError("valid_to must be on or after valid_from")
        for name in ("contribution_rate", "admin_fee_rate", "insurance_fee_rate", "asset_fee_rate"):
            _validate_rate(getattr(self, name), name)
        if self.retirement_age < 0 or self.retirement_age > 130:
            raise ValueError("retirement_age must be in [0, 130]")
        if not self.disclaimer:
            raise ValueError("disclaimer is required")

    def model_dump_jsonable(self) -> dict[str, str | int | bool]:
        return {
            "jurisdiction": self.jurisdiction,
            "version": self.version,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat(),
            "contribution_rate": str(self.contribution_rate),
            "admin_fee_rate": str(self.admin_fee_rate),
            "insurance_fee_rate": str(self.insurance_fee_rate),
            "asset_fee_rate": str(self.asset_fee_rate),
            "retirement_age": self.retirement_age,
            "official": self.official,
            "disclaimer": self.disclaimer,
        }


_POLICIES: tuple[DatedPolicyParams, ...] = (
    DatedPolicyParams(
        jurisdiction="research.cl",
        version="research-2026-10",
        valid_from=date(2020, 1, 1),
        valid_to=date(2030, 12, 31),
        contribution_rate=Decimal("0.10"),
        admin_fee_rate=Decimal("0.012"),
        insurance_fee_rate=Decimal("0.002"),
        asset_fee_rate=Decimal("0.0025"),
        retirement_age=65,
        official=False,
        disclaimer=(
            "Research fixture for method tests; not official current-law AFP rules, "
            "not legal advice, and not suitable for administration."
        ),
    ),
)


def get_policy(jurisdiction: str, as_of: date, version: str | None = None) -> DatedPolicyParams:
    if not jurisdiction:
        raise PolicyError("jurisdiction is required")
    matches = [
        p
        for p in _POLICIES
        if p.jurisdiction == jurisdiction and (version is None or p.version == version)
    ]
    if not matches:
        raise PolicyError(f"unsupported jurisdiction/version: {jurisdiction!r}/{version!r}")
    for policy in matches:
        if policy.valid_from <= as_of <= policy.valid_to:
            return policy
    raise PolicyError(f"no valid policy fixture for {jurisdiction!r} on {as_of.isoformat()}")
