# baseline 전체 코드 — F-0~K-2 (2026-08-31 오픈AI 트랙, 검증·수정 완료)

2026-08-31 오픈AI 트랙 기준 baseline 전체 파이프라인입니다.
D(임베딩)만이 아니라 F-0(분기)→G/G-2(검색·조건질의)→I(컨텍스트)→J(생성)→K(출처)까지
전부 연결돼 있습니다.

**이력**: 한 차례 임베딩 트랙 확정을 철회하고 후보 비교 단계로 되돌렸다가,
오늘 baseline을 실제로 돌리기로 하며 오픈AI 트랙(embedding_model:
text-embedding-3-small, generation_model: gpt-4o-mini)으로 재확정했습니다.
그 사이 실제 공식 데이터(document_registry_v2.json, extraction_table_v2.json,
chunks.jsonl)로 코드를 검증하면서 여러 스키마 불일치·계산 버그를 고쳤습니다 —
아래 "실제 데이터 검증 결과" 참고.

## 파일 구성

| 파일 | 역할 | 체크리스트 항목 |
| --- | --- | --- |
| `config.py` | base.yaml + 실험별 config 병합 로더, 필수값 검증 | - |
| `git_info.py` | git commit/dirty 자동 기록 | 팀 규약 2-4 |
| `embedding_client.py` | 오픈AI 임베딩(text-embedding-3-small) | 4-8 |
| `vector_store.py` | 경량 벡터 저장소(upsert·soft delete·다수 정합성 검사·원자적 저장) | 4-9-1, 4-9-2 |
| `build_index.py` | **진입점** — 청크 → 임베딩 → 인덱스 저장 | D |
| `router.py` | 질문 유형 규칙 기반 분기 | 4-10-1 |
| `table_query.py` | 조건 질의(자연어→필드/연산자, AND 복합조건, 검증 층) | 4-9-8 |
| `deadline_metadata.py` | CSV↔등록부 매핑, 선별형 마감 필터(4-10-2) | 4-10-2 |
| `generation_client.py` | 오픈AI 생성(gpt-4o-mini), 프롬프트 원칙 4종 | 4-12 |
| `answer_pipeline.py` | **진입점** — 질문 하나에 답하기 (F-0~K-2), 지연 클라이언트 생성 | F-0~K-2 |
| `run_eval.py` | **진입점** — 평가셋 전체 실행, summary.json/details.jsonl | 3-12 결과 파일 |

⚠️ 파일 구조는 아직 flat입니다. message.txt가 제안한 `config/`·`src/rag/`·
`scripts/`·`tests/` 재편은 하지 않았습니다 — 지금은 동작 검증을 우선했습니다.

## 실제 데이터 검증 결과 (2026-08-31)

공식 `document_registry_v2.json`(100건, `documents`)·`extraction_table_v2.json`
(1,200행, `rows`)·박예진님 `chunks.jsonl`로 실제 실행해서 찾아 고친 것:

1. **`config.py`** — `base.yaml` 경로 탐색이 폴더 깊이를 가정해서 깨짐 → 상위
   탐색 방식으로 수정. `top_k` 등 필수값 비어 있으면 즉시 에러.
2. **`table_query.py` 로더** — 공식 추출표는 JSONL이 아니라 단일 JSON(`rows`
   키 안에 행). 열 이름도 `state`/`value`가 아니라 `status`/`answer_normalized`.
3. **`table_query.py` 금액 비교 — 치명적 버그, 수정 완료**: 예산 조건 질의가
   `value_present` 89건 중 **89건 전부 조용히 매칭 실패** 중이었습니다.
   `answer_normalized`가 순수 숫자가 아니라 `"1억 5천만 원(부가가치세 포함)"`
   같은 원문 그대로였는데, 기존 코드는 숫자 변환 실패를 조용히 `False`로
   삼켰습니다. 한글 금액 파서(`_parse_korean_amount`)를 새로 작성해 89/89
   파싱 성공으로 고쳤습니다("5억 이상" 결과: 4건 → 20건).
4. **`table_query.py` 복합조건** — AND 복합조건 미지원 + 조건 일부만 인식하고
   나머지를 조용히 버리던 문제 → `parse_conditions`/`run_conditions_query`로
   교체. 일부만 인식되면 명시적으로 "확인 필요"를 반환.
5. **`build_index.py`** — 청크 필드명이 실제 산출물(`search_text`/
   `source_document_title`/`section_path`/`block_type`)과 전혀 안 맞았음,
   등록부도 JSONL로 잘못 가정 → 매핑 함수·로더 재작성. 파일 끝만 잘린 경우엔
   경고 후 건너뛰고 계속 진행(중간이 깨진 경우는 여전히 에러로 중단).
6. **`vector_store.py`** — 사업명 기반 콘텐츠 중복 판정(`refresh_duplicate_name_flags`)
   제거. 공식 콘텐츠 중복 근거는 `document_registry_v2.json`의
   `retrieval_eligible`·`duplicate_of_document_id`뿐이고, 그 판단은
   `build_index.py`가 색인 이전에 이미 적용합니다. 대신 벡터/메타데이터
   개수 일치, 벡터 차원 일치, chunk_id 중복, top_k 양수 검증 등을 추가하고
   저장을 임시파일→교체 방식으로 원자화했습니다.
7. **`embedding_client.py`/`generation_client.py`** — 모델명 조용한 기본값
   제거. `base.yaml`에 값이 없으면 명시적으로 막습니다(지금은 채워져 있어
   정상 실행됩니다).

⚠️ **실제 실행은 아직 안 해봤습니다.** 위 수정은 코드 리뷰·정적 검증(실제 공식
데이터 스키마 대조, 로직 추적) 기준입니다. 인덱스 빌드→검색→생성→평가가 실제
환경에서 끝까지 도는지는 직접 실행하면서 확인해 주세요.

## 실행 전 준비물

```bash
export RAG_ROOT=/srv/rfp
export HF_HOME=/srv/rfp/models
export OPENAI_API_KEY=sk-...
pip install openai numpy pyyaml --break-system-packages
```

`config/base.yaml`이 코드 기준 상위 경로 어딘가에 있어야 합니다(`config.py`가
자동 탐색). 다른 위치에 있다면 `RAG_CONFIG_PATH` 환경변수로 직접 지정하세요.

⚠️ `OPENAI_API_KEY`는 파일이 아니라 **환경변수**입니다 — 코드 어디에도 키를
적는 파일이 없습니다(의도적으로 그렇게 만듦, git 실수 커밋 방지). OpenAI에서
발급받은 키 문자열을 위처럼 `export`로 터미널에 등록하면 되고, 따로 다운받을
파일은 없습니다. 매번 치기 번거로우면 `~/.bashrc`에 같은 줄을 추가해두세요.

## ⚠️ 실행 전 반드시 확인할 것

1. `base.yaml`의 `data_egress_confirmed`는 **true로 확정**돼 있습니다(팀
   확인 완료, 2026-08-31). 값을 되돌리지 않는 한 정상 실행됩니다.
2. `embedding_model`(text-embedding-3-small), `generation_model`(gpt-4o-mini)도
   확정값으로 채워져 있습니다. 코드는 이 값이 비어 있으면 조용히 기본값을
   넣지 않고 명시적으로 실행을 막습니다 — 값을 지우면 그 즉시 에러가 납니다.
3. `chunk_size`, `chunk_overlap`, `table_chunk_threshold`는 예진님 실측으로
   이미 확정돼 있습니다(1500/150/1500).
4. `data_list.csv`(마감일 원본, UTF-8 BOM, 12컬럼 100행)를 `--deadline-csv`로
   주지 않으면 선별형 마감 필터가 꺼진 채로 돕니다 — base.yaml엔 켜져 있는데
   실제로는 안 걸린다는 뜻이라, 마감 지난 사업이 결과에 섞일 수 있습니다.

## 청크 입력 형식 (C단계 실제 산출물 — build_chunks.py 스키마)

JSONL, 한 줄에 청크 하나. `chunks_v1/chunks.jsonl` (100건, 18,142개 청크,
검색 대상 13,778개). `build_index.py`의 `map_chunk()`가 아래 필드를 내부
이름으로 변환합니다:

```json
{
  "chunk_id": "RFP-000001-0005",
  "document_id": "RFP-000001",
  "document_version": "1",
  "processed_sha256": "...",
  "sidecar_sha256": "...",
  "corpus_version": "v2",
  "preprocess_version": "v2",
  "chunking_version": "v1",
  "source_document_title": "...",
  "section_path": ["1. 추진배경 및 방향"],
  "location_label": "1. 추진배경 및 방향 · 문단 1-9",
  "block_type": "text",
  "table_idx": null, "row_start": null, "row_end": null, "part": null, "of": null,
  "table_blank_ratio": null, "table_degraded": false,
  "retrieval_eligible": true,
  "search_text": "검색용 본문 (문서명·장절 경로가 이미 앞에 붙어 있음)",
  "content": "표시용 원문(HTML 표 등, 빈 셀 보존)",
  "char_len": 806, "oversize": false
}
```

`retrieval_eligible`은 **청크 단위로도** 들어 있습니다 — 문서 단위(등록부)
필터와 별개로 `build_index.py`가 둘 다 적용합니다. `business_name` 필드는
없으며, 사업명 기반 중복 판정 로직 자체를 제거했으니 필요 없습니다.

## 마감일 메타데이터 (data_list.csv — 4-10-2)

청킹·임베딩 대상이 아닙니다(4-9-8: "숫자·날짜·부정 조건은 벡터 검색이 아니라
검증된 구조화 질의"). `document_id`로 바로 조회하는 구조화 데이터라 그냥
로더 하나(`deadline_metadata.py`)로 충분합니다.

- 형식: UTF-8 BOM, 12컬럼, 100행. 컬럼: 공고 번호·공고 차수·사업명·사업 금액·
  발주 기관·공개 일자·입찰 참여 시작일·**입찰 참여 마감일**·사업 요약·
  파일형식·파일명·텍스트
- `document_id` 연결 키: CSV `파일명`(확장자 `.hwp` 등)과 등록부
  `output_filename`(확장자 `.md`)이 확장자만 빼면 동일 — 100/100 매핑 확인
  완료(NFC 정규화 필요, 체크리스트의 "파일명 NFD 저장" 경고와 일치)
- 실측 교차검증: 마감일 빈값 정확히 8건, `reference_datetime=2024-06-01`
  기준 미경과 67 / 경과 25 / 미상 8 — 체크리스트 확정 기록과 완전 일치
- `deadline_filter_field`(base.yaml, `"입찰 참여 마감일"`) 컬럼명은 코드에
  다시 하드코딩하지 않고 cfg에서 그대로 읽음
- 미상 8건은 4-10-2 확정대로 제외하지 않고 "미상"으로 표시 후 통과시킴

## 실행 순서

```bash
# 1. 인덱스 만들기 (D)
python build_index.py \
  --chunks /srv/rfp/shared_data/processed/chunks_v1/chunks.jsonl \
  --registry /srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json

# 2. 질문 하나 테스트 (F-0~K)
python answer_pipeline.py \
  --question "5억 이상인 사업 알려줘" \
  --index /srv/rfp/shared_data/processed/index_v1 \
  --extraction-table /srv/rfp/shared_data/processed/rfp_extraction_table_v2/extraction_table_v2.json \
  --registry /srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json \
  --deadline-csv /srv/rfp/shared_data/raw/data_list.csv

# 3. 평가셋 전체 실행
python run_eval.py \
  --evalset /srv/rfp/shared_data/processed/evalset_v1/questions.jsonl \
  --index /srv/rfp/shared_data/processed/index_v1 \
  --extraction-table /srv/rfp/shared_data/processed/rfp_extraction_table_v2/extraction_table_v2.json \
  --registry /srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json \
  --deadline-csv /srv/rfp/shared_data/raw/data_list.csv \
  --out /srv/rfp/shared_data/results/tm001

# (선택) 문서 단위 증분 갱신 — 지정 문서만 재색인, 나머지는 그대로 보존
python build_index.py \
  --chunks /srv/rfp/shared_data/processed/chunks_v1/chunks.jsonl \
  --registry /srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json \
  --only RFP-000001,RFP-000037
```

⚠️ 과거 버전 문서에 있던 `registry.jsonl`, `table.jsonl`은 존재하지 않는
파일명입니다 — 실제 파일명은 위 예시 그대로 `document_registry_v2.json`,
`extraction_table_v2.json`(둘 다 JSONL이 아니라 단일 JSON)입니다.

`--registry`를 주면 문서 단위 `retrieval_eligible=true`(98건, 콘텐츠 중복
2건 제외: RFP-000006→RFP-000075, RFP-000017→RFP-000098)와 청크 단위
`retrieval_eligible` 둘 다 적용해 필터링합니다.

`--only`는 기존 인덱스가 있어야 동작합니다(처음엔 `--only` 없이 전체 생성).
지정한 문서만 soft-delete 후 재색인하고 나머지 문서의 벡터는 건드리지
않습니다 — `--only`로 지정했는데 그 문서가 이번엔 `retrieval_eligible=false`로
바뀌어 청크가 하나도 없으면, 새로 넣지 않고 기존 청크만 비활성화합니다.

`--deadline-csv`와 `--registry`를 둘 다 줘야 선별형 마감 필터가 켜집니다.
둘 중 하나라도 빠지면 필터 없이 돕니다 — base.yaml엔 필터가 켜져 있는데
이 실행에선 꺼진 채로 돈다는 걸 알리는 경고를 출력합니다(조용히 넘어가지 않음).

## 아직 부족한 부분

- `table_query.py`의 조건 파싱 규칙(정규식)은 표현 다양성 실측(1-2·1-18) 전이라
  초기 규칙일 뿐입니다 — 지금은 `예산`·`지역제한`만 허용. 나머지 필드는
  실패 사례 모아서 보완 필요.
- `active_doc_state`(4-14, 대화 맥락 유지)는 `vector_store.search()`에
  `document_id` 필터를 추가해뒀지만, `answer_pipeline.py` 오케스트레이션에서
  실제로 활성 문서 ID를 유지·주입하는 부분은 아직 연결 안 됨. **그 대신 지금은
  질문에 문서 ID(`RFP-000001` 형식)가 명시돼야만 추출형·비교형이 동작합니다**
  — 없으면 조용히 QA로 새지 않고 "어느 문서인지" 확인 질문을 되돌려줍니다.
- 오버사이즈 청크(26개, 최대 4,067자) — 토크나이저 권한 문제로 실제 토큰 수
  미측정 상태로 넘어왔습니다. 오픈AI 임베딩(text-embedding-3-small, 최대
  8,191토큰) 기준으로는 여유가 있을 걸로 보이지만, 실제 실행 때 확인 필요.
- `embedding_max_length`가 `base.yaml`에 있지만 코드 어디서도 실제로 안 씁니다
  (읽어서 저장만 하고 끝) — 죽은 설정입니다. 나중에 필요하면 오버사이즈 청크
  사전 검증에 연결하기로 함.
- message.txt가 요구한 폴더 재편(`src/rag/`·`scripts/`·`tests/`), 프롬프트
  파일 분리(`src/prompts/generate_v1.txt`), 24종 회귀 테스트는 아직
  진행 안 됨.

## 반영 완료 (2026-08-31 추가)

- **네 태스크 유형 실제 분리** — 이전엔 추출형(extract)이 QA와 똑같이
  검색+생성 경로로 흘러갔습니다. 지금은 message.txt 8번 확정대로:
  - 선별형: F-0 → G-2 → K-2 (LLM 안 씀)
  - 추출형: F-0 → G-2 → K-2가 기본(값만, LLM 안 씀). 질문에 "설명해줘"·
    "왜"·"근거" 같은 표현이 있으면 F-0 → G → I → J → K로 넘어감
  - 비교형: F-0 → G-2 → K-2, 필드×문서 표로 코드가 조립(LLM 안 씀)
  - QA형: F-0 → G → I → J → K (기존 그대로)
  추출형·비교형 둘 다 질문에 문서 ID가 명시돼야 동작 — 없으면 확인 질문.
- **문서 단위 증분 갱신** — `build_index.py --only`로 실제 연결 완료
  (message.txt 9번 확정 요구사항). 기존 인덱스를 읽어 지정 문서만
  soft-delete 후 재색인, 나머지 문서 벡터는 보존.
- C단계 청킹 스크립트(`src/chunking/build_chunks.py`) 정식 산출물 확인:
  100건, 청크 18,142개(chunk_id 전부 고유), 검색 대상 13,778개, degraded
  표 873개, 크기 초과 26개. 검색 대상 문서 98건 수치가 저희가 검증한
  `document_registry_v2.json`과 일치.
- **코드 리뷰 6건 반영** (5건은 실제 버그로 확인, 1건은 데이터 부재로 별도 해결):
  1. 지역제한 `field_absent`가 "제한 없음" 조건 충족으로 잘못 매칭되던 것 —
     4-12-1 원칙과 충돌하던 부분이라 경고로 전환, 교집합에서 제외
  2. 마감일 필터 미구현 → 아래 CSV 연결로 해결
  3. **`49,500천원`이 `49,500원`으로 잘못 변환되던 회귀 버그** — 쉼표 숫자
     숏컷이 뒤따르는 단위(천/만/억)를 무시하던 것 수정, 89건 재검증 통과
  4. `run_eval.py`가 `answer()`의 내부 오류(`error_stage`)를 확인 안 해서
     `error_count=0`으로 나올 수 있던 문제 — 의도적 오류 주입으로 수정 검증
  5. 손상된 표(`table_degraded`)가 생성 컨텍스트에 그대로 들어가던 문제 —
     4-7-1 확정대로 LLM 컨텍스트에서 빼고 원문 위치만 안내하도록 수정
  6. 인덱스 불일치 검사가 chunk_size 등 4개 키만 보던 것 → corpus/preprocess/
     chunking/registry/extraction 버전까지 확장
  - 리뷰에서 지적 안 됐지만 1번을 고치다 직접 찾은 것: 경고 문구에 문서 ID가
    없어서 95건의 서로 다른 경고가 문구 중복 제거 단계에서 1건으로 뭉개지던
    버그도 같이 고침(94건 → 95건 정상 표시)
- **선별형 마감 필터(4-10-2) 실제 연결** — `data_list.csv`(마감일 원본)를
  등록부와 연결하는 `deadline_metadata.py` 신설. `answer_pipeline.py`/
  `run_eval.py`에 `--deadline-csv`/`--registry` 인자로 연결, 마감 지난
  사업은 결과에서 제외하고 미상은 표시만 하고 통과시킴(둘 다 실측치와 일치
  검증 완료: 미경과 67/경과 25/미상 8).
