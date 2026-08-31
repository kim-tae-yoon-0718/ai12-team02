# Provenance — 6자산 + 재현성

## 6자산 (`models.PROVENANCE_ASSETS`)

평가 결과 하나가 어떤 자산 조합에서 나왔는지 **정확히 6개 필드**로 못박는다. 필드명·값 규약
(`v1`/`v2` …)은 팀 실험 인프라 `base.yaml §① "재료 버전 6칸 (필드명 고정 — 절대 개명 금지)"`
과 **정확히 일치**.

| # | 필드 | 자산 | 채우는 사람 | 현재 기본값 (`configs/default.yaml`) |
| --- | --- | --- | --- | --- |
| ① | `corpus` | 원문 코퍼스 | 박예진 | `UNKNOWN` (VERSION.txt/base.yaml 있으면 `v2`) |
| ② | `preprocess` | 전처리 파이프라인 (파싱·정제·청킹) | 박예진 | `UNKNOWN` (실데이터 연결 시 `v2`) |
| ③ | `table` | 구조화 추출 테이블 (1-12-2) | 박예진 | `UNKNOWN` (실데이터 연결 시 `v2`) |
| ④ | `index` | 검색 인덱스 (임베딩·색인 = 평가 대상 RAG, 4번) | 이태민 | **`v1`** (base.yaml §① 확정 — 폴더 v1로 시작) |
| ⑤ | `evalset` | 평가셋 (Ground Truth) | 임현진 | `UNKNOWN` (임현진 평가셋 동결 시 확정) |
| ⑥ | `scorer` | 채점기 (이 저장소) | 김하루 | **`v1`** (2026-08-31 확정) |

## 규칙

- 값이 없으면 필드를 빼지 않고 `"UNKNOWN"` (★누락 ≠ 미상 — 누락이면 회귀 3-17이 그 축을
  "안 바뀜"으로 오독).
- `EvaluationResult.provenance` 는 기본값이 빈 `Provenance` 라 절대 비지 않는다.
- `report.json` 의 `manifest.provenance` 와 `per_item.jsonl` 의 각 문항 `provenance` 가
  같은 출처(`runner.provenance()`)를 쓴다 — 어긋날 수 없다.

## 주입 순서

`--corpus/--preprocess/--table/--index/--evalset/--scorer` (CLI)
> `configs/default.yaml` `provenance:`
> `$RAG_ROOT/evalset/v1/VERSION.txt` (corpus·evalset 자동 조회)

`VERSION.txt` 포맷 (팀 확정 2026-08-30):
```
evalset: v1
corpus: [대기]
created: 2026-08-30
```

## 6칸 밖 — manifest 부가 정보

- `judge_prompt_versions` — 심판 프롬프트 파일별 버전. `scorer` 축이 움직였을 때
  "코드가 바뀐 건가 프롬프트가 바뀐 건가"를 가르는 재료 (오염 방지 4-9).
- `git_commit` / `git_dirty` — 실행 진입점에서 자동 기록 (규약 §2-4).
  **`git_dirty=true` 면 그 실행은 재현 불가** — 최종 실험은 false 여야 한다.

## 회귀 원인 분리 (`diagnostics.regression.attribute_change`, 3-17)

두 실행의 `manifest.provenance` 를 6축 그대로 diff. `changed_keys` / `clean_comparison`
(한 번에 하나만 바뀌었는가) / `<asset>_changed_flag`.

## 오염 방지 (`diagnostics.contamination`, 4-9)

`scorer` 축만 바뀌었는데 target_metric 이 올랐으면 **오염 의심** → "시스템 개선"으로 인정 안 함.
실험은 `target_metrics` / `protected_metrics` / `expected_direction` 을 선언해야 한다.
