from __future__ import annotations

from datetime import date
from decimal import Decimal
from pprint import pprint

from afp_operations import demo
from afp_operations.ops import ContributionEvent, InMemoryLedger, PensionQuoteRequest


def main() -> None:
    ledger = InMemoryLedger()
    for month in range(1, 4):
        ledger.record_contribution(
            ContributionEvent(
                member_id="example-anonymous-member",
                paid_at=date(2026, month, 28),
                taxable_wage=Decimal("1200.00"),
                jurisdiction="research.cl",
            )
        )
    quote = ledger.quote_pension(
        PensionQuoteRequest(
            member_id="example-anonymous-member",
            survival=[Decimal("1.00"), Decimal("0.97"), Decimal("0.93"), Decimal("0.88")],
            discount_rate=Decimal("0.02"),
        )
    )
    pprint(
        {
            "package_demo": demo(),
            "three_month_reconciliation": ledger.reconcile(
                "example-anonymous-member"
            ).model_dump_jsonable(),
            "research_quote": quote.model_dump_jsonable(),
        }
    )


if __name__ == "__main__":
    main()
