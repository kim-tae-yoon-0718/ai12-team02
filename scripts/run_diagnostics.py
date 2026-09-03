#!/usr/bin/env python3
"""3-1-0 진단표 — `python -m grader.cli diagnose ...` 래퍼.

  scripts/run_diagnostics.py --report artifacts/scores/report.json
"""
import sys

from grader.cli import main

if __name__ == "__main__":
    sys.argv = ["grader", "diagnose", *sys.argv[1:]]
    raise SystemExit(main())
