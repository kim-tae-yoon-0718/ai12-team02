# rfp_extraction_table_v4 (공식)

공식 `rfp_extraction_table_v3`의 1,200행·12필드 구조를 유지하면서 원문 판정이
잘못되었던 두 행을 정정한 공식 구조화 추출표다.

| 바뀐 행 | v3 | v4 |
|---|---|---|
| RFP-000067 / 컨소시엄 요건 | `field_absent` | `value_present` (PMR-007, md L1701) |
| RFP-000081 / 컨소시엄 요건 | `field_absent` | `value_present` (PMR-02 md L765 + PMR-05 md L795) |

- RFP-000067은 공동수급으로 제안할 때 역할과 책임을 정의하라는 조건이 있다.
- RFP-000081은 하도급이 전체 사업금액의 10%를 넘을 때 공동수급체를 구성하라는
  조건이 있다. 따라서 무조건적인 공동수급 필수 문서는 아니다.
- 스키마 버전은 `1-12-2/v3`로 유지한다. 열 구조가 아니라 두 행의 의미 판정만
  바뀌었기 때문이다.
- 후보본 `v3.1-candidate.1`을 팀 버전 규칙에 따라 정수 버전 `v4`로 승격했다.
- 후보 재생성 당시 v3과 1,200행 전체 열을 대조했으며, 위 두 행 외 변화는 0건이었다.

파일:

- `extraction_table_v4.json` / `extraction_table_v4.csv`: 공식 실행 자료
- `extraction_metadata.json`: 버전·입력·생성 기록
- `semantic_decisions_v4.csv`: 의미 판정 원본
- `field_alias_candidates_v4.csv`: 필드 별칭 검토 자료
