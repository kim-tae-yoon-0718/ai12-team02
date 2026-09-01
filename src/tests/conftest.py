"""pytest 실행 시 src/rag·src/scripts를 sys.path에 추가한다 —
test_baseline.py가 `from table_query import ...`(rag 모듈)뿐 아니라
`import run_eval`(scripts 진입점)도 바로 import할 수 있게.
tests/conftest.py는 pytest가 자동으로 읽는 파일이라 별도 설정 없이도 적용된다."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "rag"))
sys.path.insert(0, str(_SRC / "scripts"))
