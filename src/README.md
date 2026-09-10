# RFP baseline — 실행 가이드

RFP(제안요청서) 문서에 대해 선별형·추출형·QA형·비교형 네 가지 질문 유형을
분기(F-0)해서 각각 다른 경로로 답하는 baseline입니다. 오픈AI 트랙
(`text-embedding-3-small`, `gpt-5-mini`)으로 확정돼 있습니다.

**핵심 원칙**: 값이 구조화 자료(추출표 v5 · identity_v2)에 있으면 LLM한테 다시
쓰게 하지 않고 코드로 그대로 조립합니다. LLM은 원문 설명이 필요한 경우에만
태우고, 그때도 **구조화 값은 확정 값으로 못박아** 프롬프트에 넣습니다.

## 공식 입력 자료 (읽기 전용)

| 무엇 | 어디 | 쓰는 곳 |
|---|---|---|
| 청크 v3 | `$RAG_ROOT/shared_data/processed/chunks_v3/chunks.jsonl` | 일반 QA 원문 근거 |
| 문서 등록부 v2 | `.../document_registry_v2/document_registry_v2.json` | 검색 대상·중복 판정, 해시 대조 |
| **identity_v2** | `.../document_registry_v2/document_identity_v2.csv` | **마감일(`bid_deadline`)·발주기관·사업명** |
| 추출표 v5 | `.../rfp_extraction_table_v5/extraction_table_v5.json` | 12필드 값·상태·위치 |
| 추출표 이름표 | `.../rfp_extraction_table_v5/extraction_metadata.json` | 위 표의 공식 메타데이터 |

> ⚠️ 마감일은 **identity_v2가 유일한 출처**입니다(2026-09-02, 4-4 확정).
> 예전 코드가 읽던 원본 수집 CSV 경로는 폐기했습니다. identity_v2에 값이
> 없으면 다른 자료로 조용히 대체하지 않고 "미상"으로 답합니다.

## 폴더 구조

```
저장소 최상위/
├── config/
│   └── base.yaml
└── src/
    ├── rag/          라이브러리 모듈
    │   ├── config.py            경로·버전 조립
    │   ├── text_normalize.py    기관명·사업명 비교용 정규화
    │   ├── identity_metadata.py identity_v2 로더(마감일·기관·사업명)
    │   ├── doc_resolver.py      문서 특정 우선순위
    │   ├── table_query.py       조건 파서 + 추출표 로드·검증
    │   ├── vector_store.py      경량 벡터 저장소
    │   ├── embedding_client.py / generation_client.py
    │   ├── pricing.py           비용 계산(100만 토큰 단가)
    │   ├── router.py            F-0 분기
    │   └── git_info.py
    ├── scripts/      실행 진입점
    │   ├── build_index.py
    │   ├── answer_pipeline.py
    │   ├── run_eval.py
    │   ├── real_api_smoke_test.sh
    │   └── run_practice_evalset.sh
    ├── prompts/      generate_v1.txt
    └── tests/        conftest.py, test_baseline.py, test_pr10_fixes.py,
                      test_official_inputs.py
```

## 실행 전 준비 (한 번만)

```bash
source /srv/rfp/venv/bin/activate
export RAG_ROOT=/srv/rfp
export HF_HOME=/srv/rfp/models
export OPENAI_API_KEY=sk-...   # 코드는 값을 읽어 보관하지 않고 존재 여부만 확인합니다
```

## 질문 하나 넣기

```bash
cd src/scripts
python3 answer_pipeline.py --question "5억 이상인 사업 알려줘"
```

`--index` / `--extraction-table` / `--extraction-metadata` / `--identity` /
`--registry`를 안 넘기면 `config/base.yaml`의 버전값으로 `config.py`가 경로를
자동 조립합니다. 명시적으로 주면 그 값이 항상 우선합니다.

| 자동 조립 | base.yaml 값 | 실제 예시 |
|---|---|---|
| 인덱스 | `index` | `$RAG_ROOT/shared_data/processed/index_v2` |
| 추출표 | `table` | `.../rfp_extraction_table_v5/extraction_table_v5.json` |
| 추출표 이름표 | `table` | `.../rfp_extraction_table_v5/extraction_metadata.json` |
| 등록부 | `document_registry_version` | `.../document_registry_v2/document_registry_v2.json` |
| identity_v2 | `document_registry_version` | `.../document_registry_v2/document_identity_v2.csv` |
| 청크 | `chunking_version` | `.../chunks_v3/chunks.jsonl` |

**추출표와 identity_v2는 필수입니다.** 못 찾으면 경고가 아니라 에러로 멈춥니다.
테스트 목적의 우회는 명시적 옵션으로만 가능합니다:

- `--allow-unofficial-table` — 공식 규모(1,200행·100문서·12필드)·버전 검증 생략
- `--allow-no-identity` — identity_v2 없이 실행
- `--allow-no-deadline-filter` — 마감 필터가 켜져 있는데 자료가 없을 때 통과

### 공식 인덱스

현재 활성 인덱스는 **`index_v2`** 입니다(`base.yaml`의 `index: v2`).

| | 경로 | 내용 |
|---|---|---|
| 활성 | `$RAG_ROOT/shared_data/processed/index_v2` | chunks_v3 · registry_v2 · extraction v3 · `text-embedding-3-small`(1536차원) · 98문서 / 14,828청크 |
| 보존 | `$RAG_ROOT/shared_data/processed/index_v1` | chunking v1 · extraction v2 — **낡음**. 실행 코드가 차단하지만 기록으로 남겨 둡니다(삭제·수정 금지) |

`--index` 를 주지 않으면 `base.yaml` 의 `index` 값으로 공식 `index_v2` 가 자동 선택됩니다.

### 인덱스 만들기

```bash
python3 build_index.py \
  --chunks   $RAG_ROOT/shared_data/processed/chunks_v3/chunks.jsonl \
  --registry $RAG_ROOT/shared_data/processed/document_registry_v2/document_registry_v2.json \
  --index-out /path/to/index_dir        # 생략하면 base.yaml의 index 경로
```

인덱싱 전에 다음을 **전부** 검사하고, 하나라도 어긋나면 중단합니다:

- 청크의 `corpus_version` / `chunking_version` 존재 및 설정 일치
- `chunks_v3/VERSION.txt`(공식 메타데이터)와 교차 확인 — 청크끼리 값이 같다는
  것만으로는 최신 자료로 인정하지 않습니다
- 등록부 대조: `document_id` · `document_version` · `processed_sha256` ·
  `sidecar_sha256` · 검색 대상 여부 · 중복 제외 관계
- 등록부 없는 일반 실행은 실패(`--allow-no-registry`로만 우회)

### 평가셋 전체 돌리기

```bash
python3 run_eval.py --evalset /srv/rfp/evalset/practice_items.jsonl \
                    --out /path/to/결과폴더
```

## 4단계 검증 순서

```bash
# 1. 비용 없는 자동 테스트 (OpenAI 실호출 0건)
cd src && python -m pytest tests chunking -v

# 2. 실제 API로 1문항만
cd src/scripts && ./real_api_smoke_test.sh

# 3. 연습용 8문항 전체
cd src/scripts && ./run_practice_evalset.sh /path/to/결과폴더

# 4. 김하루님 채점기로 responses.jsonl 채점
#    ⚠️ 채점기 실행 파일이 아직 서버 공유 경로에 없습니다(base.yaml scorer: null)
```

## 질문 유형과 실제 실행 경로(route)

| task_type | 기본 경로 | LLM 호출 |
|---|---|---|
| select(선별형) | 추출표 조건 질의 | 안 함 |
| extract(추출형) | 추출표/identity_v2 단건 조회 | 기본은 안 함, "설명해줘"류면 검색+생성 |
| compare(비교형) | 추출표 필드×문서 조립 | 안 함 |
| qa(QA형) | 벡터 검색 + 생성 (+ 구조화 자료 결합) | 함 |

⚠️ **`task_type`(평가셋 라벨)과 실제 `route`(내부 경로)는 다를 수 있습니다.**
이건 정상 동작이고, 채점에서 그 자체로 감점되지 않습니다(2026-09-02 확인).
`route`는 기록용입니다.

**route 값 9가지** (`answer_pipeline.ALL_ROUTES`와 동일):

| route | 뜻 |
|---|---|
| `추출테이블_문서선별` | 선별형 |
| `추출테이블_값조회` | 추출형 12필드 기본 경로 |
| `추출테이블_비교조립` | 비교형 |
| `chunks검색_LLM답변` | 검색+생성(구조화 자료 없음) |
| `구조화자료_결합_LLM답변` | 구조화 확정 값 + 청크 근거를 함께 넣어 생성 |
| `identity_v2_값조회` | 마감일 단건 조회 |
| `애매_되묻기` | 확인 질문 / 거절 |
| `검색불필요_인사응답` | 인사 |
| `검색불필요_사용법안내` | 시스템 사용법 안내 |

## 문서 특정 정책 (2026-09-02 확정)

우선순위:

1. 질문에 명시된 정확한 문서 ID (`RFP-000001`)
2. 이전 대화에서 확정된 `active_document_id` — **후속 질문일 때만**
   (지시 표현이 있거나, 다른 이름 단서가 전혀 없을 때)
3. 정규화한 발주기관명 + 사업명이 **둘 다** 일치
4. 정규화한 발주기관명만 일치
5. 정규화한 사업명만 일치하며 후보가 정확히 1개

모호한 경우 — 후보 0개면 "찾을 수 없음", 1개면 선택, **2개 이상이면 후보를
보여주며 되묻습니다**(임의로 첫 문서를 고르지 않습니다).

**기관명 정규화**(비교할 때만, 원본 표시는 보존):
NFKC → `(사)`/`사단법인`, `(재)`/`재단법인`, `(주)`/`㈜`/`주식회사` 제거 →
문장부호·괄호·공백 제거 → casefold.
`협회`·`연구원`·`진흥원`처럼 기관의 핵심 이름은 **절대 지우지 않습니다**.

**사업명 정규화**: NFKC + 공백/문장부호 정리까지만. 단어를 임의로 빼거나
LLM·임베딩으로 "비슷한 사업"을 자동 선택하지 않습니다. 다만 `project_name`
앞에 같은 행의 `buyer_org`가 글자 그대로 붙어 있는 경우, 그 접두부만 떼어낸
형태도 후보로 봅니다(결정적 연산, 데이터 근거 있음).

CLI 단발 실행에서 후속 질문을 테스트하려면:

```bash
python3 answer_pipeline.py --question "거기 예산 얼마야" --active-document RFP-000001
```

## QA형에서 구조화 자료를 함께 쓰는 규칙

QA형으로 분류돼도 질문에 확정된 구조화 필드가 있으면 구조화 자료를 함께 씁니다.

```
질문 분류는 QA형 유지 → 마감일/12필드 질문인지 확인 → 공식 구조화 값 먼저 조회
→ 설명이 필요하면 chunks_v3도 검색 → 구조화 값 + 원문 설명을 함께 넣어 답변
```

- 마감일 질문 → `identity_v2`의 값과 위치를 우선 사용
- 12필드 질문 → `extraction_table_v5`의 값·상태·위치를 우선 사용
- 프롬프트에서 구조화 값은 "공식 확정 값"으로, 청크는 "설명·문맥 보완 근거"로
  구분해 넣고, 모델이 청크를 보고 확정 값을 바꾸지 못하게 지시합니다
- 단, 구조화 자료 자체가 `conflict`면 하나를 고르지 않고 충돌로 답합니다

**추출표 상태별 처리**

| status | 답변 |
|---|---|
| `value_present` | 확정 값 사용 |
| `field_absent` | "원문에 항목 자체가 없음"(값이 없다고 단정하지 않음) |
| `external_reference` | "외부 공고문·붙임을 직접 확인" |
| `not_disclosed` | "비공개로 명시됨" |
| `conflict` | 서로 다른 내용과 **위치를 모두** 제시하고 기권 |
| `extraction_failed` / `review_required` | 정상 값처럼 답하지 않고 기권 |

## 조건 질의 (선별형)

- 숫자 안의 쉼표는 천 단위 구분자로 보호한 **뒤에** 조건을 나눕니다
  (`49,500`이 `49`와 `500`으로 쪼개지지 않습니다)
- 연산자: `이상`→`>=`, `초과`/`넘는`→`>`, `이하`/`이내`→`<=`, `미만`→`<`
  — 네 연산자 모두 실제 금액까지 파싱합니다
- 복합 단위(`1억 5천만원`)를 한 덩어리로 읽습니다
- 지역제한은 **뜻으로** 판정합니다. 빈칸(`field_absent`)을 "제한 없음"으로 보지
  않고, `value_present`라는 이유만으로 "제한 있음"으로도 보지 않습니다

## `run_eval.py`가 만드는 결과 파일

- `details.jsonl` — 내부 진단용(라우팅 근거, 세션, 오류 단계, 문항별 비용)
- `summary.json` — 전체 통계 + 실제 사용한 입력 경로
- `api_cost_summary.json` — 요청 수·토큰·단가·비용·실행시간(문항별 포함)
- **`responses.jsonl`** — 채점기가 읽는 파일. 한 줄=문항 하나:

```json
{
  "id": "PRAC-EXT-001",
  "answer": "최종 답변 문장",
  "structured_answer": null,
  "contexts": [],
  "retrieved": [],
  "citations": [{"document": "RFP-000038", "section": "2. 사업목표",
                 "ref_no": "paragraph 3 · line 61", "line": 61,
                 "block_type": "paragraph", "block_index": 3,
                 "source": "extraction_table_v3"}],
  "selected_document_ids": [],
  "abstained": false,
  "route": "추출테이블_값조회",
  "failure": null,
  "latency_ms": 812,
  "cost_usd": 0.0
}
```

- 최상위 구조는 **질문 유형과 무관하게 항상 같습니다**
- `structured_answer` — 선별형: `["RFP-000001", ...]` / 비교형:
  `{"RFP-000038": {"예산": {...}}}` / 추출·QA형: 필드 상태 객체
- `contexts`/`retrieved` — 실제 LLM에 넣은 청크 / 검색된 후보 전체.
  `document_id`·`section_path`·`section_paths`·`block_type`·`block_index`·
  `md_line_start`·`md_line_end`·`location_label`·`search_text`를 보존합니다
  (텍스트 청크의 `block_index`도 비우지 않습니다)
- `citations` — 추출표의 `representative_location`과 `additional_locations`를
  **모두** 기록합니다. 비교형은 문서×필드마다, 선별형은 조건마다 남깁니다

## 비용 계산 (`cost_usd`)

단가 단위는 **100만(1M) 토큰당 달러**이고 키 이름에 단위가 박혀 있습니다.

```yaml
pricing_unit: "per_1m_tokens"
pricing:
  gpt-5-mini:
    input_per_1m: 0.25
    cached_input_per_1m: 0.025
    output_per_1m: 2.0
  text-embedding-3-small:
    input_per_1m: 0.02
```

- 문항 시작마다 사용량을 초기화합니다 — 앞 문항 비용이 다음 문항으로 복사되지
  않습니다
- 추출표·identity_v2만 쓴 문항은 **생성 API 비용이 0**입니다
- 생성 비용과 임베딩 비용을 나눠서 기록합니다

## GPT-5 Mini 요청 인자

`gpt-5-mini`는 다음 인자를 지원하지 않아 요청에 **아예 싣지 않습니다**:
`temperature`, `top_p`, `logprobs`, `max_tokens`.
대신 `max_completion_tokens`(시작값 4096, `base.yaml`에서 변경 가능)를 씁니다.

> ⚠️ GPT-5 Mini에서는 API 기본값을 사용하며 동일 문장의 완전한 재현은 보장하지
> 않습니다. `base.yaml`의 `temperature: null`은 "보내지 않음"이라는 뜻입니다.

## 안전장치 (조용히 안 넘어가고 막는 것들)

- 공식 추출표·identity_v2를 못 찾으면 → 에러 (`--allow-*` 옵션으로만 우회)
- 추출표가 1,200행·100문서·문서당 12필드·확정 필드명·선언 버전과 다르면 → 에러
- 청크 버전이 없거나 설정·공식 메타데이터와 다르면 → 에러
- 청크와 등록부의 버전·해시·문서 집합이 다르면 → 인덱스 생성 중단
- 마감 필터가 켜져 있는데 자료를 못 찾으면 → 에러
- `chunks.jsonl`이 중간에 잘려 있으면 → 에러
- 단일 질문 처리 오류 → 종료코드 1
- 평가 문항 오류 1건 이상 또는 빈 평가셋 → 종료코드 1 (`--allow-errors`로만 우회)
- 문항당 자동 재시도는 최대 1회 (`--max-item-retries`, 2 이상은 거부)
- 오류 문구에서 API 키·이메일·전화번호를 마스킹합니다
- `session_id` 없는 평가셋은 문항마다 독립. 이어붙이려면 `--continuous-session`

## 존재하지 않는 사업 판별

유사도 임계값 대신 구조화 데이터로 판단합니다. 질문에 기관명처럼 생긴 표현이
있는데 identity_v2 기관 목록에 없으면 검색·생성을 안 태우고 "찾을 수 없음"으로
답합니다. 알려진 기관과 알 수 없는 기관이 함께 있어도 알 수 없는 쪽을 놓치지
않습니다(공식 87개 기관 전수 오탐 0건 확인).

## 아직 안 끝난 것

- 김하루님 채점기 실행 파일이 서버 공유 경로에 없음 → `base.yaml` `scorer: null`
- 마감 지난 사업 "목록" 조회 기능 — 보류
- 청킹 표 파싱 문제(파이프 표, 중첩 표) — 예진님 담당, 재작업 중
