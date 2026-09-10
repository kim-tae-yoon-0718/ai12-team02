# 입찰메이트

> 공공 RFP 100건을 대상으로 **조건에 맞는 공고 선별**, **정확한 값 추출**, **근거 기반 질의응답**을 수행하는 하이브리드 RAG 시스템입니다.

## 시연 영상

<video src="https://github.com/user-attachments/assets/9f28c66a-3748-45df-93ef-a931f3886f43" controls width="100%"></video>

데모에서는 다음 기능을 사용할 수 있습니다.

- **질문하기:** RFP 조건 검색, 값 추출, 문서 비교, 근거 기반 설명
- **회사 매칭:** 회사 정보를 바탕으로 입찰 후보를 우선 검토·확인 필요·마감 임박으로 구분

## 무엇을 해결하나요?

일반적인 RAG의 top-k 검색은 질문과 비슷한 문서 조각을 찾는 데 강하지만, 검색 대상 전체를 빠짐없이 검사했다고 보장하기 어렵습니다. 예를 들어 “예산이 5억 원 이상인 사업을 모두 보여줘”라는 질문에 상위 몇 개 문서만 검색하면 조건에 맞는 공고를 놓칠 수 있습니다.

입찰메이트는 질문에 따라 일을 나눕니다.

| 질문 종류 | 처리 방식 |
|---|---|
| 전체 문서의 정형 조건 선별 | 100문서 × 12필드 추출표를 코드로 전수 계산 |
| 특정 문서의 값 추출·비교 | 검증된 구조화 값을 코드가 직접 조립 |
| 배경·목적처럼 문맥이 필요한 질문 | 원문 청크 검색 후 gpt-5-mini가 근거 기반 답변 생성 |
| 불확실하거나 지원 범위를 벗어난 질문 | 임의로 답하지 않고 LLM 안전망에서 재판단하거나 되묻기 |

## 주요 기능

| 기능 | 질문 예시 | 결과 |
|---|---|---|
| 전체 조건 선별 | “예산 5억 원 이상인 사업을 모두 보여줘” | 조건을 만족하는 공고 목록 |
| 특정 값 추출 | “RFP-000038의 예산과 공동수급 조건은?” | 추출값과 근거 |
| 명시된 문서 비교 | “RFP-000038과 RFP-000049의 예산과 기간을 비교해줘” | 문서 × 필드 비교 |
| 근거 기반 QA | “이 사업의 추진 배경을 설명해줘” | 검색 근거를 포함한 설명 |
| 모호성 처리 | “그 사업의 제출 방식은?” | 활성 문서 확인 또는 추가 질문 |
| 회사 맞춤 추천 | 회사의 지역·분야·자격 입력 | 지원 후보와 확인할 조건 |

## 빠른 시작

### 1. 저장소 받기

~~~bash
git clone https://github.com/kim-tae-yoon-0718/ai12-team02.git
cd ai12-team02
~~~

### 2. 실행 환경 만들기

Python 3.10 이상이 필요합니다.

~~~bash
python -m venv .venv
~~~

macOS·Linux:

~~~bash
source .venv/bin/activate
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

의존성 설치:

~~~bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install openai numpy gradio
~~~

### 3. 실행 자료 배치

원본 RFP와 대용량 청크·벡터 파일은 저장소에 포함하지 않습니다. 전체 질의응답과 데모를 실행하려면 아래 공식 자료가 필요합니다.

| 자료 | 기본 버전 | 기본 위치 |
|---|---|---|
| 전처리 코퍼스 | v2 | $RAG_ROOT/shared_data/processed/corpus_v2 |
| 검색 청크 | v3 | $RAG_ROOT/shared_data/processed/chunks_v3 |
| 문서 등록부·문서 식별 정보 | v2 | $RAG_ROOT/shared_data/processed/document_registry_v2 |
| 벡터 인덱스 | v2 | $RAG_ROOT/shared_data/processed/index_v2 |
| 추출표 | v5 | data/preprocessed/rfp_extraction_table_v5 |
| 평가셋 | v3 | data/evalsets/final/v3/items.jsonl |
| 채점기 | v3 | config/grader.yaml |

자료의 생성·검증 방법은 [데이터 안내](data/README.md)를 확인해 주세요.

### 4. 환경변수 설정

macOS·Linux:

~~~bash
export RAG_ROOT=/srv/rfp
export OPENAI_API_KEY="YOUR_API_KEY"
~~~

Windows PowerShell:

~~~powershell
$env:RAG_ROOT="C:\path\to\rfp"
$env:OPENAI_API_KEY="YOUR_API_KEY"
~~~

API 키를 저장소, 로그, 결과 파일에 기록하지 마세요.

### 5. 질문 실행

~~~bash
python src/scripts/answer_pipeline.py \
  --question "예산이 5억 원 이상인 사업을 모두 보여줘"
~~~

### 6. 데모 실행

~~~bash
python demo/app.py
~~~

터미널에 표시되는 로컬 주소를 브라우저에서 열면 됩니다.

## 처리 구조

<p align="center">
  <img src="docs/images/stage3-hybrid-routing.svg" width="100%" alt="규칙 고속 경로와 LLM 안전망을 결합한 입찰메이트 처리 구조">
</p>

1. 질문을 구조화해 문서·도구·조건·필드를 정합니다.
2. 결과가 규칙으로 완전히 확정되는 질문은 **고속 경로**가 추출표를 조회하고 답을 직접 조립합니다.
3. 문맥 검색이나 예외 판단이 필요한 질문은 **LLM 안전망**이 원문과 도구 결과를 확인합니다.
4. 두 경로 모두 출처·누락·출력 형식을 확인한 뒤 답변을 반환합니다.

기본 설정은 [config/base.yaml](config/base.yaml)에서 확인할 수 있습니다.

## 평가와 검증

### 현재 개발용 50문항 결과

평가셋 v3·채점기 v3·추출표 v5 기준입니다.

| 전체 | 선별형 | 추출형 | QA형 | 만점 문항 | API 비용 | 실행 시간 |
|---:|---:|---:|---:|---:|---:|---:|
| **0.9400** | **1.0000** | **1.0000** | **0.7000** | 47/50 | 약 $0.0693 | 4분 7초 |

### 3단계 고속 경로의 효과

동일한 평가셋 v2·채점기 v2·추출표 v4로 2단계－1·2·3차와 비교했습니다.

| 항목 | 2단계－1·2·3차 | 3단계 |
|---|---:|---:|
| 전체 점수 | 0.8000 | **0.8000** |
| 고속 경로 처리 | 0/50 | **32/50** |
| LLM 안전망 처리 | 50/50 | **18/50** |
| API 비용 | 약 $0.2486 | **약 $0.1219** |
| 실행 시간 | 18분 7초 | **8분 31초** |

점수는 유지하면서 API 비용을 **50.96%**, 실행 시간을 **52.98%** 줄였습니다.

> **수치 해석 주의:** 0.8000은 모델 구조만 비교하기 위해 평가셋 v2·채점기 v2·추출표 v4를 고정한 실험 결과입니다. 이후 추출표의 일부 값 범위와 평가 정답이 서로 다르거나, 같은 값을 요구하면서도 출력 표기가 달라 오답으로 처리되던 부분을 원문 기준으로 맞췄습니다. 이 정렬을 반영한 평가셋 v3·채점기 v3·추출표 v5의 현재 개발 점수가 0.9400입니다. 따라서 0.8000에서 0.9400으로 오른 전체 차이를 모델만의 향상으로 해석하지 않습니다. QA형에는 개발용 문자열 기반 판정이 포함되어 있어 사람의 최종 의미 평가를 대신하지 않습니다.

2단계－1·2·3차의 시간과 비용은 처음 실행한 50문항 중 답변 출력 방식을 고친 추출형 8번과 11번만 다시 실행해 교체한 합산값입니다. 즉, 최초 실행의 두 문항 비용·시간을 빼고 수정 후 재실행값을 더했으며, 3단계와 같은 시점에 나란히 실행한 A/B 결과는 아닙니다.

### 50문항 실행

~~~bash
python src/scripts/run_eval.py \
  --evalset data/evalsets/final/v3/items.jsonl \
  --out outputs/run50
~~~

### 개발용 채점

~~~bash
grader run \
  --evaluation-set data/evalsets/final/v3/items.jsonl \
  --responses outputs/run50/responses.jsonl \
  --mode development \
  --corpus v2 --preprocess v2 --table v5 \
  --index v2 --evalset v3 --scorer v3 \
  --out-dir outputs/score
~~~

### 전체 테스트

~~~bash
python -m pytest
~~~

pyproject.toml은 모델·청킹·채점기 테스트를 함께 수집하도록 설정되어 있습니다.

## 안전장치

- 필요한 공식 자료가 없으면 다른 버전으로 조용히 대체하지 않고 중단합니다.
- 문서 후보가 여러 개면 임의로 첫 문서를 선택하지 않습니다.
- 추출표 값이 충돌하면 하나를 고르지 않고 근거 위치와 함께 알립니다.
- 문항 오류와 API 차단을 정상 답변으로 숨기지 않습니다.
- 실행 결과에 사용한 자료 버전, 경로, 비용과 오류 단계를 기록합니다.

## 알려진 제한

- 여러 문서 결과를 “그중”, “1번”, “예”처럼 계속 좁히는 후속 대화는 안정적으로 지원하지 않습니다. 활성 문서 한 건을 같은 실행에서 이어 묻는 경우만 제한적으로 지원합니다.
- 추출표의 12개 필드 밖에서 98개 문서 전체를 훑어 자유로운 문자열 OR·NOT 조건을 조합하는 전수 검색 도구는 아직 없습니다.
- 지원하지 않는 질문을 고속 경로가 드물게 완료로 판단하거나, LLM 안전망이 같은 질문에 다른 해석을 낼 수 있습니다.
- 청크 v4·인덱스 v3는 안전성 개선 후보이며 현재 기본 실행에는 연결하지 않았습니다.

## 개발에 참여하기

1. 최신 main에서 작업 브랜치를 만듭니다.
2. 기능과 함께 테스트를 추가합니다.
3. python -m pytest로 전체 회귀를 확인합니다.
4. 설정·자료 버전을 바꿨다면 새 버전을 만들고 기존 버전을 덮어쓰지 않습니다.
5. 변경 이유, 검증 결과, 영향 범위를 PR에 작성합니다.

공식 평가셋·추출표·인덱스 같은 고정 자료를 수정해야 한다면 모델 변경과 분리하고, 버전과 파일 내용으로 만든 디지털 지문을 함께 남겨 주세요.

## 저장소 구조

~~~text
ai12-team02/
├── config/                    # 공식 설정과 실험별 설정
├── data/
│   ├── evalsets/final/v3/     # 개발용 평가셋
│   └── preprocessed/          # 구조화 추출표
├── demo/                      # Gradio 데모
├── docs/                      # 구조·스키마·채점·검증 문서
├── src/
│   ├── chunking/              # 문서 청킹
│   ├── grader/                # 평가·채점 파이프라인
│   ├── rag/                   # 검색·라우팅·도구 실행
│   └── scripts/               # 질문·평가·인덱스 실행 진입점
├── tests/                     # 채점기·통합 테스트
└── tools/company_match/       # 회사 정보 기반 입찰 후보 추천
~~~

## 기술 스택

- **Language:** Python 3.10+
- **Generation:** OpenAI gpt-5-mini
- **Embedding:** OpenAI text-embedding-3-small, 1,536차원
- **Vector search:** NumPy 기반 경량 저장소, cosine similarity
- **Schema & config:** Pydantic, PyYAML
- **Demo:** Gradio
- **Test & evaluation:** Pytest, 자체 평가셋과 다층 채점기

## 관련 문서

- [응답 계약](docs/response_contract.md)
- [채점 방식](docs/scoring.md)
- [검증 규칙](docs/validation.md)

## 팀원별 작업일지

| 팀원 | 주요 담당 | 작업일지 |
|---|---|---|
| 박예진 | 문서 전처리, 구조 보존 청킹, 인덱스 생성·검증 | 업로드 예정 |
| 김태윤 | 구조화 추출표, 모델 실험, 최종 경로 통합 | [작업·실험일지](https://app.notion.com/p/3c19556c393780bfa177d2a3065208a6) |
| 임현진 | 평가셋 구축, 데모 구현·통합 | [작업일지](https://app.notion.com/p/3c7c6fc04696811eb166db9e257d1c96) |
| 김하루 | 다층 채점기, 출처·기권·회귀 진단 | [작업일지](https://app.notion.com/p/3c75532982f380b8a948c1f3809c78f9) |
| 이태민 | 검색·생성 베이스라인, 회사 정보 기반 공고 추천 | [작업일지](https://splendid-dewberry-887.notion.site/3d696d9eeef280e88ac7d2630c44a6f7?source=copy_link) |

## 협업일지

- [팀 협업일지](https://app.notion.com/p/3c69556c393780ec80dfcd249a76bd0d)
