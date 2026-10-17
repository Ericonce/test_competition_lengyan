"""Convert labeled competition inputs to the class folders used by training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls.data.conversion import convert_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Source YAML, e.g. configs/data/label_file.yaml")
    parser.add_argument("--output", required=True, help="New directory for the converted class folders")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    converted = convert_dataset(args.data, args.output, progress=not args.no_progress)
    print(f"Conversion complete. Train with data={str(converted)!r}")


if __name__ == "__main__":
    main()
