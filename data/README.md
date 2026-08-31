# data/ — 로컬 데이터 (내용물은 커밋 금지, NDA)

`.gitkeep` 만 커밋되고 실제 파일은 `.gitignore` 로 막혀 있다. 각자 로컬에 채운다.

| 디렉토리 | 무엇 | 출처 |
| --- | --- | --- |
| `raw/` | 원본 RFP 문서 (HWP/PDF) | 박예진 (체크리스트1) |
| `preprocessed/` | 전처리 결과 (md/sidecar) | 박예진 — 공식본은 `$RAG_ROOT/shared_data/processed/` |
| `evalsets/practice/` | practice 세트 (`practice_items.jsonl`, CI용) | 임현진 |
| `evalsets/final/` | 최종 50문항 | 임현진 (누수 방지 — 이 저장소엔 안 들어온다) |
| `gold/` | 추출 테이블 감사 등 기준 데이터 | 박예진/태윤 |
| `outputs/` | 시스템(4번) 응답 JSONL | 이태민 |

스키마 확인용 최소 샘플은 `tests/fixtures/` 에 있다.
