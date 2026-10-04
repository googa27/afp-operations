from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from afp_operations import capabilities, demo
from afp_operations.bayes import smoke_inference


def _dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="afp-operations")
    sub = parser.add_subparsers(dest="command", required=True)
    commands: dict[str, Callable[[], dict[str, Any]]] = {
        "capabilities": capabilities,
        "demo": demo,
        "bayes-smoke": smoke_inference,
    }
    for name in commands:
        sub.add_parser(name)
    args = parser.parse_args(argv)
    _dump(commands[args.command]())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
