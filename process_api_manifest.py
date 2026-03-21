from __future__ import annotations

import argparse
from pathlib import Path

from scripts.pipeline import build_manifest_from_text_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build api_manifest.csv from apis.txt")
    parser.add_argument("input", nargs="?", default="apis.txt", type=Path, help="Path to apis.txt")
    parser.add_argument("output", nargs="?", default="api_manifest.csv", type=Path, help="Path to output CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_manifest_from_text_input(args.input.resolve(), args.output.resolve())
    print(f"written: {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
