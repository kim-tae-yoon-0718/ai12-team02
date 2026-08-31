# rag_grader_pipeline — 팀 공용 단축 명령
#
#   make            도움말
#   make install    의존성 설치 (개발용 포함)
#   make test       pytest -q
#   make lint       ruff 검사
#   make data       박예진 산출물(/srv/rfp) → grader 입력 재생성  ← RAG_ROOT 필요
#   make check      실데이터로 1·2층 검사 (make data 를 먼저 돌림)
#   make clean      캐시·생성물 정리
#
# data/ 아래 산출물은 .gitignore 대상(NDA). 커밋하지 않고 `make data` 로 매번 다시 만든다.

RAG_ROOT ?= /srv/rfp
GOLD_DIR  := data/gold
PY        := python3
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH),)

.DEFAULT_GOAL := help
.PHONY: help install test lint data check clean

help:
	@sed -n '3,12p' $(MAKEFILE_LIST)

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

# ── 박예진 산출물(/srv/rfp) → grader 입력 (자동 재생성) ─────────────────
# corpus_doc_ids.json / excluded_doc_ids.json  (참조 무결성 검사용)
# data_manifest.json                            (CI 1층 데이터 정상성용)
data:
	@test -d "$(RAG_ROOT)" \
		|| { echo "RAG_ROOT($(RAG_ROOT)) 가 없다. export RAG_ROOT=/srv/rfp 후 다시."; exit 1; }
	RAG_ROOT=$(RAG_ROOT) $(PY) scripts/build_doc_ids.py --out-dir $(GOLD_DIR)
	RAG_ROOT=$(RAG_ROOT) $(PY) scripts/build_data_manifest.py --out $(GOLD_DIR)/data_manifest.json
	@echo "→ $(GOLD_DIR)/ 재생성 완료"

# ── 실데이터 1층(데이터 정상성) 스모크 ────────────────────────────────
# 실제 50문항 평가셋이 오기 전이라 평가셋 자리는 픽스처로 둔다(스키마만 확인).
# 1층은 /srv/rfp 실데이터로 검증된다. 2층 참조 무결성은 실제 평가셋이 오면
#   make check EVALSET=data/evalsets/final/final.jsonl REFCHECK='--corpus-doc-ids $(GOLD_DIR)/corpus_doc_ids.json --excluded-doc-ids $(GOLD_DIR)/excluded_doc_ids.json'
EVALSET  ?= tests/fixtures/evaluation_set.jsonl
REFCHECK ?=
check: data
	RAG_ROOT=$(RAG_ROOT) $(PY) -m grader.cli run --mode checks \
		--evaluation-set $(EVALSET) \
		--data-manifest $(GOLD_DIR)/data_manifest.json $(REFCHECK)

clean:
	rm -rf .pytest_cache .cache .ruff_cache artifacts/scores/* $(GOLD_DIR)/*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
