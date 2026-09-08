# data/ — 데이터 사용 규칙

원본 RFP와 대용량 전처리 자료는 계속 저장소에 올리지 않고 각자 로컬 또는
공유 서버에서 사용한다.

**예외 1 — `gold/` 아래 3개 파일은 커밋한다** (문서 본문 없음, 참조 무결성 검사 공유용):
`corpus_doc_ids.json` · `excluded_doc_ids.json` · `data_manifest.json`.
`RAG_ROOT=/srv/rfp make data` 로 재생성되며, 문서 등록부가 바뀌면 갱신해 다시 커밋한다.

**예외 2 — 팀 개발용 공식 골든셋은 버전별로 커밋한다.**
현재 공식본은 `evalsets/final/v3/items.jsonl`이며 `evalsets/final/CURRENT`가
현재 버전을 가리킨다. 새 버전을 만들 때 기존 버전은 지우지 않는다.
향후 처음 보는 데이터로 성능을 한 번만 측정할 숨은 최종 평가셋은 이 폴더와 분리한다.

**예외 3 — 실행과 채점의 기준이 되는 공식 구조화 추출표는 버전별로 커밋한다.**
현재 공식본은 `preprocessed/rfp_extraction_table_v5/`이다. v4는 저장소에 그대로
보존하며, 새 실행은 v5를 사용한다. v5는 v4와 같은 12필드 구조이고 원문 대조가
끝난 6개 행의 누락 정보·목록 경계·정리값만 정정했다.

| 디렉토리 | 무엇 | 출처 |
| --- | --- | --- |
| `raw/` | 원본 RFP 문서 (HWP/PDF) | 박예진 (체크리스트1) |
| `preprocessed/` | 전처리 결과와 공식 구조화 추출표 v5 | 박예진/팀 검토 — 대용량 원문은 `$RAG_ROOT/shared_data/processed/` |
| `evalsets/practice/` | practice 세트 (`practice_items.jsonl`, CI용) | 임현진 |
| `evalsets/final/` | 팀 개발용 공식 골든셋 50문항(버전 관리) | 임현진/팀 검토 |
| `gold/` | 추출 테이블 감사 등 기준 데이터 | 박예진/태윤 |
| `outputs/` | 시스템(4번) 응답 JSONL | 이태민 |

스키마 확인용 최소 샘플은 `tests/fixtures/` 에 있다.
