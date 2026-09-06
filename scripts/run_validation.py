#!/usr/bin/env python3
"""평가셋 스키마·무결성 검사 — `python -m grader.cli validate ...` 래퍼.

  scripts/run_validation.py --evaluation-set data/evalsets/final/final.jsonl --strict
"""
import sys

from grader.cli import main

if __name__ == "__main__":
    sys.argv = ["grader", "validate", *sys.argv[1:]]
    raise SystemExit(main())
