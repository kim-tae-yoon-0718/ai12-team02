#!/usr/bin/env python3
"""회귀 판정·변동 폭 리포트 — `python -m grader.cli regression ...` 래퍼.

  변동 폭:  scripts/generate_report.py --runs r1.json r2.json r3.json --out artifacts/regression/variance.json
  회귀 판정: scripts/generate_report.py --baseline prev.json --current now.json --variance artifacts/regression/variance.json
"""
import sys

from grader.cli import main

if __name__ == "__main__":
    sys.argv = ["grader", "regression", *sys.argv[1:]]
    raise SystemExit(main())
