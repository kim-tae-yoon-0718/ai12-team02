"""테스트는 저장소 루트 기준 상대 경로(config/grader.yaml, tests/fixtures/...)를 쓴다.
어디서 pytest 를 돌리든 그 경로가 맞도록 루트로 chdir 한다."""
import os
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _chdir_repo_root():
    old = os.getcwd()
    os.chdir(_ROOT)
    try:
        yield
    finally:
        os.chdir(old)
