from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stream_policy.questionnaire.analysis import analyze_responses


def run(path: Path) -> dict[str, object]:
    return analyze_responses(pd.read_csv(path.resolve(strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
