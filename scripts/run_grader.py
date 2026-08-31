#!/usr/bin/env python3
"""채점 실행 — `python -m grader.cli run ...` 의 얇은 래퍼.

  scripts/run_grader.py --evaluation-set tests/fixtures/evaluation_set.jsonl \
      --responses tests/fixtures/model_responses.jsonl --mode development
"""
import sys

from grader.cli import main

if __name__ == "__main__":
    sys.argv = ["grader", "run", *sys.argv[1:]]
    raise SystemExit(main())
