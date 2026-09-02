# RFP baseline — 실행 가이드

RFP(제안요청서) 문서에 대해 선별형·추출형·QA형·비교형 네 가지 질문 유형을
분기(F-0)해서 각각 다른 경로로 답하는 baseline입니다. 오픈AI 트랙
(text-embedding-3-small, gpt-5-mini)으로 확정돼 있습니다.

**핵심 원칙**: 값이 구조화된 표(추출표·CSV)에 있으면 LLM한테 다시 쓰게
하지 않고 코드로 그대로 조립합니다(선별형·추출형·비교형은 기본적으로
LLM 미호출). LLM은 원문 설명이 진짜 필요한 경우(QA형, 또는 "설명해줘"가
붙은 추출형)에만 태웁니다.

## 폴더 구조

```
저장소 최상위/
├── config/
│   └── base.yaml
└── src/
    ├── rag/          라이브러리 모듈(config.py, vector_store.py 등)
    ├── scripts/      실행 진입점 3개(build_index.py, answer_pipeline.py, run_eval.py)
    ├── prompts/      generate_v1.txt
    └── tests/        conftest.py, test_baseline.py
```

## 실행 전 준비 (한 번만)

```bash
source /srv/rfp/venv/bin/activate
export RAG_ROOT=/srv/rfp
export HF_HOME=/srv/rfp/models
export OPENAI_API_KEY=sk-...   # 팀 운영진한테 받은 키. 발급받은 문자열이지 다운로드하는 파일이 아님
```

`~/.bashrc`에 export 세 줄을 추가해두면 새 터미널마다 안 쳐도 됩니다.

## 질문 어떻게 넣나 — 가장 간단한 형태

```bash
cd src/scripts
python3 answer_pipeline.py --question "5억 이상인 사업 알려줘"
```

**`--question`만 있으면 됩니다.** `--index`/`--extraction-table`/`--registry`/
`--deadline-csv`를 매번 안 넘겨도, `config/base.yaml`에 적힌 버전값
(`corpus`, `table`, `index` 등)으로 `config.py`가 실제 경로를 자동으로
조립합니다:

| 자동 조립되는 것 | base.yaml 값 기준 | 실제 예시 |
|---|---|---|
| 인덱스 | `index` | `$RAG_ROOT/shared_data/processed/index_v1` |
| 추출표 | `table` | `$RAG_ROOT/shared_data/processed/rfp_extraction_table_v2/extraction_table_v2.json` |
| 문서 등록부 | `corpus`(또는 `document_registry_version`) | `$RAG_ROOT/shared_data/processed/document_registry_v2/document_registry_v2.json` |
| 마감일 CSV | 고정 파일명(`data_list.csv`) | `$RAG_ROOT/shared_data/raw/data_list.csv` |

특정 경로를 강제로 지정하고 싶으면(예: 실험용 인덱스 따로 테스트) 명시적으로
주면 그게 항상 우선합니다:

```bash
python3 answer_pipeline.py --question "질문" --index /path/to/다른/인덱스
```

**자동 조립이 실패해도(예: RAG_ROOT 미설정) 인덱스 말고는 안 죽습니다** —
추출표/등록부/마감일CSV는 못 찾으면 경고만 찍고 그 기능만 꺼진 채로
계속 진행합니다. 인덱스는 필수라 못 찾으면 명확한 에러로 멈춥니다.

### 평가셋 전체 돌리기

```bash
python3 run_eval.py --evalset /path/to/questions.jsonl --out /path/to/결과폴더
```
`--evalset`, `--out`은 매번 다르니 그대로 필수고, 나머지 경로는 위와
동일하게 자동 조립됩니다.

## 4단계 검증 순서

```bash
# 1. 비용 없는 자동 테스트 (OpenAI 실호출 0건)
cd src
pytest tests/test_baseline.py -v

# 2. 실제 API로 1~2문항만 (scripts/real_api_smoke_test.sh 참고)
# 3. 연습셋 전체 (scripts/run_practice_evalset.sh 참고)
# 4. 하루님 평가 인프라로 채점 — 아래 responses.jsonl 참고
```

## 네 가지 질문 유형과 실제 실행 경로(route)

| task_type | 기본 경로 | LLM 호출 |
|---|---|---|
| select(선별형) | 추출표 조건 질의 | 안 함 |
| extract(추출형) | 추출표 단건 조회 | 기본은 안 함, "설명해줘"류 붙으면 검색+생성으로 전환 |
| compare(비교형) | 추출표 필드×문서 조립 | 안 함 |
| qa(QA형) | 벡터 검색 + 생성 | 함 |

⚠️ **`task_type`(평가셋 라벨)과 실제 `route`(내부 경로)는 다를 수 있습니다.**
예를 들어 평가셋엔 `qa`로 라벨된 문항이 "얼마야?" 같은 단순 값 질문이면
저희 시스템은 더 안전한 추출형 경로(`추출테이블_값조회`)로 답합니다.
**이건 정상 동작입니다** — 하루님 채점기 정책상 `route`가 `task_type`과
달라도 그 자체로 오답 처리되지 않는다고 확인됐습니다(2026-09-02). `route`는
기록용이고, 최종 답이 맞는지만 채점에 씁니다.

route 값 5가지:
| route | 뜻 |
|---|---|
| `추출테이블_문서선별` | 선별형 |
| `추출테이블_값조회` | 추출형 기본 경로 |
| `추출테이블_비교조립` | 비교형 |
| `chunks검색_LLM답변` | 검색+생성(QA, 설명요청 추출형) |
| `애매_되묻기` | 확인질문/거절 |
| `검색불필요_인사응답` | 인사 등 부가 경로 |

## 문서 특정 방식 (추출형·비교형·QA형 공통)

우선순위: ① 질문에 명시된 문서 ID(`RFP-000001`) → ② 발주기관명(부분일치,
못 찾거나 모호하면 정식 명칭으로 재요청) → ③ "그 사업"/"거기"류 지시
표현 + 세션에 남은 직전 활성 문서.

CLI 단발 실행에서 anaphora("거기 예산 얼마야" 등)를 테스트하려면
`--active-document`로 직접 주입:
```bash
python3 answer_pipeline.py --question "거기 예산 얼마야" --active-document RFP-000001
```

## `run_eval.py`가 만드는 결과 파일

- `details.jsonl` — 내부 진단용(라우팅 근거, 세션 상태, 오류 단계 등)
- `summary.json` — 전체 통계
- **`responses.jsonl`** — 하루님 채점기가 읽는 파일. 한 줄=문항 하나:

```json
{
  "id": "q001",
  "answer": "최종 답변 문장",
  "structured_answer": null,
  "contexts": [],
  "retrieved": [],
  "citations": [{"document": "RFP-000001", "section": "...", "ref_no": "line 61"}],
  "selected_document_ids": [],
  "abstained": false,
  "route": "추출테이블_값조회",
  "failure": null,
  "latency_ms": 812,
  "cost_usd": null
}
```

- `structured_answer` — 선별형: `["RFP-000001", ...]` / 비교형: `{"RFP-000038": {"예산": "...", "사업기간": "..."}, ...}`
- `contexts`/`retrieved` — 실제 LLM에 넣은 청크 / 검색된 후보 전체(예진님 청크 형식 그대로)
- `cost_usd`는 `base.yaml`에 단가(`pricing.*`)를 채워야 계산됩니다 — 안 채우면 `null`

## base.yaml — cost_usd 단가 채우는 위치

`generation_model`/`embedding_model` 설정 근처, 최상위 레벨에 `pricing`
키를 새로 만들어서 넣으시면 됩니다:

```yaml
pricing:
  gpt-5-mini:
    input_per_1k: null   # TODO — $/1K input 토큰, OpenAI 최신 가격표 확인
    output_per_1k: null  # TODO — $/1K output 토큰
  text-embedding-3-small:
    input_per_1k: null   # TODO — $/1K 토큰
```

`generation_model`/`embedding_model`에 적힌 모델명과 `pricing` 아래 키
이름이 정확히 같아야 합니다(지금은 `gpt-5-mini`, `text-embedding-3-small`).
값을 안 채우면 `cost_usd`는 계속 `null`로 나옵니다 — 에러는 안 납니다.

## 안전장치 (조용히 안 넘어가고 막는 것들)

- 마감 필터가 켜져 있는데(`deadline_filter_default.select`) CSV·등록부를
  못 찾으면 → 에러. `--allow-no-deadline-filter`로만 우회
- 청크에 버전·해시가 없으면 → 에러(조용히 빈 값으로 안 채움)
- `chunks.jsonl`이 중간에 잘려 있으면 → 에러
- `embedding_model`/`generation_model`이 비어 있으면 → 에러
- 프롬프트 파일(`prompts/generate_v1.txt`)이 없으면 → 에러
- 평가 문항에 오류가 있으면 → `run_eval.py`가 실패 종료코드 반환
  (`--allow-errors`로만 우회)
- `session_id` 없는 평가셋은 기본적으로 문항마다 독립(세션 안 섞임).
  이어붙이려면 `--continuous-session` 명시

## 존재하지 않는 사업 판별 (QA-004류)

유사도 임계값을 임의로 정하지 않고, 질문에 기관명처럼 생긴 표현
("OOO재단/OOO청/OOO공사" 등)이 있는데 실제 `data_list.csv` 기관 목록에
없으면 검색·생성 자체를 안 태우고 바로 "찾을 수 없음"으로 답합니다
(비용도 절감됨). 실제 87개 기관 전수로 오탐 0건 확인.

## 마감일 메타데이터

`data_list.csv`는 청킹·임베딩 대상이 아닙니다 — `document_id`로 바로
조회하는 구조화 데이터입니다. CSV `파일명`(확장자 제외)과 등록부
`output_filename`(확장자 제외)이 같은 것으로 연결됩니다.

## 아직 안 끝난 것

- `cost_usd` 실제 단가표 (팀 확인 필요, 위 참고)
- 마감 지난 사업 "목록" 조회 기능 — 보류
- 청킹 표 파싱 문제(파이프 표, 중첩 표) — 예진님 담당, 재작업 중
- 실제 OpenAI API 종단 실행 — 스크립트만 준비됨(2단계), 서버에서 확인 필요
