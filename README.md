# RAG Grader Pipeline

입찰공고 RAG 시스템을 **평가셋(Ground Truth) 기준으로 채점**하는 파이프라인.
확정 평가셋 스키마(v0.2, 임현진)를 입력 계약으로 삼고 팀 확정 채점 정책을 코드로 강제한다.

**체크리스트3 — 평가 인프라 · 담당 김하루 · 브랜치 `HR`**

---

## 팀원은 여기부터

| 알고 싶은 것 | 볼 곳 |
| --- | --- |
| 평가셋 문항을 어떻게 써야 하나 (입력 계약) | [`docs/schema.md`](docs/schema.md) · `src/grader/models.py` · `tests/fixtures/` |
| 어떻게 채점하나 (내용/형식/기권/가중치) | [`docs/scoring.md`](docs/scoring.md) |
| 검색·출처 좌표 채점 | [`docs/scoring.md`](docs/scoring.md) |
| 평가셋 자체 검증 (CI 2층) | [`docs/validation.md`](docs/validation.md) |
| 6-자산 provenance / 재현성 | [`docs/provenance.md`](docs/provenance.md) |
| 전체 구조 / 5층 실행 흐름 | [`docs/architecture.md`](docs/architecture.md) |

---

## 디렉토리

```text
Makefile          make install/test/lint/data/check — 팀 공용 단축 명령
config/           grader.yaml (채점 설정 한 곳) + ci.yaml/local.yaml (오버레이) + base.yaml (팀 공통 모델 설정)
prompts/          judge_*.v1.md — 팀 확정 심판 프롬프트 원문
docs/             설계·규칙 문서 (위 표)
scripts/          run_* (CLI 래퍼) + build_doc_ids / build_data_manifest (박예진 산출물 → grader 입력)
src/grader/       채점기 본체
  ├─ models.py        입력/출력 계약 (pydantic)  ← 팀원은 여기부터
  ├─ config.py normalize.py cache.py providers.py
  ├─ task_scoring.py retrieval.py extraction.py judge.py prompts.py   채점
  ├─ validation/      schema · answer · location · provenance · assets   평가셋 검증
  ├─ diagnostics/     statistics · report · retrieval · citation · extraction · regression · contamination
  ├─ versioning.py runner.py cli.py
data/             로컬 데이터 (NDA — .gitkeep 만 커밋, data/README.md 참고)
artifacts/        실행 산출물 (scores / diagnostics / extraction_audit / regression — 구조만 커밋)
tests/            unit/ · integration/ · regression/ · fixtures/
```

---

## 설치 · 실행

```bash
make install          # = pip install -e ".[dev]"
make test             # = pytest -q  (159 tests)
make lint             # = ruff check
make data             # 박예진 산출물(/srv/rfp) → data/gold/*.json 재생성  (RAG_ROOT 필요)
make check            # 실데이터로 1층 데이터 정상성 스모크

# 스키마만 검사
python -m grader.cli validate --evaluation-set tests/fixtures/evaluation_set.jsonl

# 값싼 검사 (1~3층, 모델 호출 없음)
python -m grader.cli run --evaluation-set tests/fixtures/evaluation_set.jsonl --mode checks

# 개발 실행
python -m grader.cli run \
  --evaluation-set tests/fixtures/evaluation_set.jsonl \
  --responses tests/fixtures/model_responses.jsonl --mode development

# CI (practice 세트 필수 — 없으면 exit 1, 임현진 2-17)
python -m grader.cli run --config config/grader.yaml --overlay config/ci.yaml \
  --evaluation-set data/evalsets/final/final.jsonl \
  --responses data/outputs/responses.jsonl --mode ci \
  --practice-set data/evalsets/practice/practice_items.jsonl

# 최종 (누수 방지: --allow-final + 실제 심판 + --extraction-audit 필수)
python -m grader.cli run ... --mode final --allow-final --extraction-audit data/gold/audit.jsonl --runner 김하루

# 진단 / 회귀
python -m grader.cli diagnose --report artifacts/scores/report.json
python -m grader.cli regression --runs r1.json r2.json r3.json --out artifacts/regression/variance.json

pytest -q          # 159 tests
```

`GRADER_PROVIDER=mock`(기본) / `openai_compatible`(+`GRADER_API_KEY`/`GRADER_MODEL`).
`use_stub_judge: true` 동안은 provider 무관하게 StubJudge(배관 점검용, `tier=final` 불가).

산출물: `report.json`(층별 결과 + 집계 + 실행 조건), `per_item.jsonl`(문항별 결과 + provenance).
