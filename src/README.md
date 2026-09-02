# RFP baseline 전체 코드 — F-0~K-2 (오픈AI 트랙)

RFP(제안요청서) 문서에 대해 선별형·추출형·QA형·비교형 네 가지 질문 유형을
분기(F-0)해서 각각 다른 경로로 답하는 baseline입니다. 오픈AI 트랙
(text-embedding-3-small, gpt-4o-mini)으로 확정돼 있습니다.

**가장 중요한 설계 원칙**: 값이 구조화된 표(추출표·CSV)에 있으면 LLM한테
다시 쓰게 하지 않고 코드로 그대로 조립합니다(선별형·추출형·비교형은
기본적으로 LLM 미호출). LLM은 원문 설명이 진짜 필요한 경우(QA형, 또는
추출형에 "설명해줘"가 붙은 경우)에만 태웁니다. 이게 지켜지는지가
`tests/test_baseline.py`의 `TestTaskTypeRoutingGeneralization`에서 가장
강하게 검증되는 부분입니다.

## 폴더 구조

```
저장소 최상위/
├── config/
│   └── base.yaml
└── src/
    ├── rag/                    라이브러리 모듈
    │   ├── config.py
    │   ├── vector_store.py
    │   ├── embedding_client.py
    │   ├── generation_client.py
    │   ├── router.py
    │   ├── table_query.py
    │   ├── deadline_metadata.py
    │   └── git_info.py
    ├── scripts/                실행 진입점 3개
    │   ├── build_index.py
    │   ├── answer_pipeline.py
    │   └── run_eval.py
    ├── prompts/
    │   └── generate_v1.txt      생성 프롬프트(코드에서 분리됨)
    └── tests/
        ├── conftest.py
        └── test_baseline.py
```

`config.py`/`generation_client.py`는 자기 파일 위치 기준으로 상위 경로를
탐색해서 `config/base.yaml`·`prompts/generate_v1.txt`를 찾습니다 — 폴더
구조가 이후에 또 바뀌어도 코드 수정 없이 그대로 동작하도록 만들어뒀습니다.

## 실행 전 준비

```bash
# 팀 공용 가상환경(한 번만)
python3 -m venv /srv/rfp/venv
source /srv/rfp/venv/bin/activate
pip install openai numpy pyyaml pytest

export RAG_ROOT=/srv/rfp
export HF_HOME=/srv/rfp/models
export OPENAI_API_KEY=sk-...   # OpenAI에서 발급받은 키. 다운로드 아니고 발급만 받으면 됨
```

⚠️ `OPENAI_API_KEY`는 파일이 아니라 **환경변수**입니다 — 코드 어디에도 키를
적는 파일이 없습니다. 매번 치기 번거로우면 `~/.bashrc`에 export 줄을
추가해두세요.

## 4단계 실행 파이프라인

코드를 실제 서버에 반영한 뒤 이 순서로 검증합니다. 뒷 단계로 갈수록 비용이
커지니, 앞 단계가 전부 통과한 뒤에만 다음으로 넘어갑니다.

### 1단계 — 비용 없는 자동 테스트 (OpenAI 실호출 0건)

```bash
cd src
pytest tests/test_baseline.py -v
```

71개 테스트, 실제 공식 데이터 스키마를 그대로 따르는 픽스처로 검증합니다.
전부 모의(mock) 응답이라 비용이 안 듭니다. 크게 두 부류입니다:

- **버그 회귀 방지** — 금액 파서, field_absent 처리, 오류 집계, 손상 표
  제외, 인덱스 정합성 검사 등 지금까지 실제로 발견해서 고친 버그들이
  다시 재발하지 않는지
- **라우팅 검증(`TestTaskTypeRoutingGeneralization`)** — 가장 중요한
  부분. 선별형·추출형·QA형·비교형이 다양한 질문 표현에서 실제로 올바른
  경로로 갈라지는지, 그리고 **LLM을 태워야 할 때만 태우는지**(선별형·
  단순추출형·비교형은 호출되면 테스트가 바로 실패하도록 만들어둠)

### 2단계 — 실제 API로 1~2문항만

```bash
bash ../scripts/real_api_smoke_test.sh
```

추출형(비용 0, G-2만 탐) + QA형(비용 소액, 진짜 임베딩+생성 호출) 딱
두 개만 실제로 돌려서 키·결제·응답 형식이 정상인지 최소 비용으로 확인.

### 3단계 — 연습셋 8문항

```bash
bash ../scripts/run_practice_evalset.sh
```

pretty-print JSON 8개를 표준 JSONL로 자동 변환한 뒤 실행합니다.
`--continuous-session`을 켜서, 문항들이 이어지는 하나의 대화(anaphora
포함)로 처리되게 합니다.

### 4단계 — 하루님 평가 인프라

`details.jsonl`을 그대로 넘기면 됩니다. 문항별로 선택한 경로·분기 근거·
검색 청크 ID·조건 질의 결과·오류 단계까지 다 들어있어서, 검색·문서특정·
생성 중 어디서 실패했는지 분리해서 볼 수 있습니다. (정확한 인터페이스는
하루님 채점기 쪽과 별도 확인 필요 — 아직 연결 확정 안 됨.)

## 네 가지 질문 유형과 경로

| 유형 | 경로 | LLM 호출 |
|---|---|---|
| 선별형(select) | F-0 → G-2(조건 질의) → K-2 | 안 함 |
| 추출형(extract), 기본 | F-0 → G-2(표 직접 조회) → K-2 | 안 함 |
| 추출형, "설명해줘"류 | F-0 → G(검색) → I → J → K | 함 |
| 비교형(compare) | F-0 → G-2(필드×문서 표 조립) → K-2 | 안 함 |
| QA형(qa) | F-0 → G(검색) → I → J → K | 함 |

**문서 특정 방식(`resolve_document_id`, `answer_pipeline.py`)** — 우선순위:
1. 질문에 명시된 문서 ID(`RFP-000001` 형식)
2. 질문에 언급된 발주기관명(org_only) — 부분일치로 시도, 못 찾거나 여러
   문서에 매칭되면 정식 명칭이나 문서 ID로 다시 요청(임의로 안 고름)
3. "그 사업"/"거기"류 지시 표현(anaphora) + 세션에 남은 직전 활성 문서

**추출형이 다루는 값 두 종류**:
- 12개 확정 필드(사업 개요·사업분야·공고일·사업기간·예산·참가자격·
  지역제한·컨소시엄 요건·평가배점·제출방식·필수제출서류·과업범위) →
  `extraction_table_v2.json`(G-2)
- 마감일 → 12필드에서 빠진 값이라 별도로 `data_list.csv`(CSV)를 직접
  조회 (`detect_deadline_question`)

**존재하지 않는 사업 판별(QA-004류)** — 유사도 임계값을 임의로 정하지
않고, 질문에 기관명처럼 생긴 표현("OOO재단/OOO청/OOO공사" 등)이 있는데
실제 `data_list.csv` 기관 목록에 하나도 없으면 검색·생성 자체를 안 태우고
바로 "찾을 수 없음"으로 답합니다(`detect_unknown_org`). 87개 실제 기관
전수로 오탐 0건 확인.

## 안전장치 — 조용히 넘어가지 않고 막는 것들

baseline 단계에서 반복적으로 "조용한 실패"가 발견돼서, 다음은 전부
경고가 아니라 **에러로 중단**하도록 확정했습니다.

- `base.yaml`에 선별형 마감 필터가 켜져 있는데(`deadline_filter_default.select`)
  `--deadline-csv`/`--registry`를 안 주면 → 에러. `--allow-no-deadline-filter`로만
  명시적 우회 가능
- 청크에 `document_version`/`processed_sha256`/`sidecar_sha256`이 없으면
  → 에러(예전엔 빈 문자열로 조용히 채웠음)
- `chunks.jsonl` 파일이 중간에 잘려 있으면 → 에러(예전엔 마지막 줄만
  스킵하고 계속 진행했는데, 조용한 데이터 손실이라 되돌림)
- `embedding_model`/`generation_model`이 비어 있으면 → 에러(기본값 자동
  선택 안 함)
- 프롬프트 파일(`prompts/generate_v1.txt`)이 없으면 → 에러(코드에
  하드코딩된 문구로 대체 안 함)
- 인덱스 불일치 검사(`config_mismatch_check`)가 corpus/preprocess/
  chunking/registry/extraction 버전까지 다 봄(예전엔 chunk_size 등 4개만)

**평가셋 세션 기본값도 안전한 쪽으로**: `session_id` 필드가 없으면
문항마다 완전히 독립(매 문항 세션 리셋)이 기본값입니다. "서로 무관한
문항인데 우연히 활성 문서가 섞이는" 게 "이어지는 문항인데 안 이어지는"
것보다 훨씬 위험한 실패라서요. 일부러 하나의 대화로 이어붙이고 싶으면
`--continuous-session`을 명시해야 합니다. 평가셋 문항 자체에
`active_document_id`가 박혀 있으면 그 값이 이어받기보다 우선합니다
(테스트 하네스가 그 문항 전용 전제조건을 직접 주입하는 경우).

## 마감일 메타데이터 (data_list.csv)

청킹·임베딩 대상이 아닙니다. `document_id`로 바로 조회하는 구조화
데이터라 `deadline_metadata.py`가 로더 역할만 합니다.

- 형식: UTF-8 BOM, 12컬럼, 100행
- `document_id` 연결 키: CSV `파일명`(확장자 `.hwp` 등)과 등록부
  `output_filename`(확장자 `.md`)이 확장자만 빼면 동일 — 100/100 매핑
  확인됨(NFC 정규화 필요)
- 실측: 마감일 빈값 8건, `reference_datetime=2024-06-01` 기준 미경과
  67 / 경과 25 / 미상 8 — 체크리스트 기존 기록과 일치
- 미상 8건은 제외하지 않고 "미상"으로 표시 후 통과

## 기관명 매칭 정책 (baseline 확정, 임시 타협)

체크리스트 4-9-5는 "기관명 매칭은 단순 규칙으로 불가(부분일치 허용/
금지 둘 다 실패 사례 있음) — 정규화 수준은 별도 실측 소관, 대기"로
남겨뒀습니다. baseline 단계에서는 다음 타협으로 확정합니다:

- 부분 일치(키워드)로 시도
- 서로 포함 관계인 기관명은 긴 쪽만 인정(`_filter_subsumed_org_names`) —
  "서울특별시" vs "서울특별시교육청" 같은 오매칭 방지. 실제 87개 기관
  전수 스캔으로 이런 쌍 3건 확인, 전부 자동으로 커버됨
- 그래도 못 찾거나 모호하면 조용히 QA로 새지 않고, 문서 ID나 발주기관
  **정식 명칭**으로 다시 말해달라고 되묻습니다(임의로 하나를 안 고름)

표현 다양성 실측 결과가 나오면 이 타협은 재검토가 필요합니다.

## 아직 결정·구현 안 된 것

- **마감 지난 사업 "목록" 조회** — "5억 이상인 사업"처럼 목록형으로
  "마감 지난 사업 뭐 있어?"를 물었을 때 지난 것만 보여주는 기능. 보류 —
  일단 baseline 먼저, 나중에 구현하기로 함
- **청킹 표 파싱 문제** — 파이프 표(`| a | b |`) 검색 텍스트 누락(1,384건),
  중첩 표 HTML 깨짐(269건). 예진님 담당, 저희 쪽 방어 코드도 안 넣기로
  결정됨(청킹 쪽에서 직접 수정 예정)
- 폴더 재편 이후 24종 회귀 테스트(message.txt 원래 요구) 중 일부는
  `test_baseline.py`로 흡수됐지만 전부는 아님
- 4단계(하루님 평가 인프라)의 정확한 입력 형식 — `details.jsonl`을
  그대로 쓰는지 별도 변환이 필요한지 미확인
- 실제 OpenAI API 종단 실행 자체는 아직 서버에서 안 해봄(스크립트만
  준비됨, 2단계 참고)

## 청크 입력 형식 (실제 build_chunks.py 산출물)

100건, 청크 18,142개, 검색 대상 13,778개. `build_index.py`의
`map_chunk()`가 실제 필드명을 내부 이름으로 변환합니다:

```json
{
  "chunk_id": "RFP-000001-0005",
  "document_id": "RFP-000001",
  "document_version": "1",
  "processed_sha256": "...", "sidecar_sha256": "...",
  "corpus_version": "v2", "preprocess_version": "v2", "chunking_version": "v1",
  "source_document_title": "...",
  "section_path": ["1. 추진배경 및 방향"],
  "block_type": "text",
  "table_idx": null, "row_start": null, "row_end": null, "part": null, "of": null,
  "table_degraded": false, "retrieval_eligible": true,
  "search_text": "검색용 본문(문서명·장절 경로가 이미 앞에 붙어 있음)",
  "content": "표시용 원문", "char_len": 806, "oversize": false
}
```

`retrieval_eligible`은 청크 단위로도 있습니다 — 문서 단위(등록부) 필터와
별개로 `build_index.py`가 둘 다 적용합니다.

## 실행 명령 모음

```bash
cd src/scripts

# 인덱스 만들기 (전체)
python3 build_index.py \
  --chunks /srv/rfp/shared_data/processed/chunks_v3/chunks.jsonl \
  --registry /srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json

# 문서 단위 증분 갱신 (지정 문서만 재색인, 나머지 벡터는 보존)
python3 build_index.py \
  --chunks .../chunks.jsonl --registry .../document_registry_v2.json \
  --only RFP-000001,RFP-000037

# 질문 하나 테스트
python3 answer_pipeline.py \
  --question "5억 이상인 사업 알려줘" \
  --index /srv/rfp/shared_data/processed/index_v1 \
  --extraction-table /srv/rfp/shared_data/processed/rfp_extraction_table_v2/extraction_table_v2.json \
  --registry /srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json \
  --deadline-csv /srv/rfp/shared_data/raw/data_list.csv

# 평가셋 전체 실행
python3 run_eval.py \
  --evalset .../questions.jsonl \
  --index /srv/rfp/shared_data/processed/index_v1 \
  --extraction-table .../extraction_table_v2.json \
  --registry .../document_registry_v2.json \
  --deadline-csv .../data_list.csv \
  --out /srv/rfp/runs/결과폴더
```

`--registry`를 주면 문서 단위 `retrieval_eligible=true`(98건, 콘텐츠
중복 2건 제외: RFP-000006→RFP-000075, RFP-000017→RFP-000098)와 청크
단위 `retrieval_eligible` 둘 다 적용합니다.
