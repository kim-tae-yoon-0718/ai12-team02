#!/usr/bin/env bash
# 3단계 검증 — 임현진님 연습용 8문항을 실제 API로 끝까지 돌린다.
#
# 사용법:
#   source /srv/rfp/venv/bin/activate
#   export RAG_ROOT=/srv/rfp
#   export OPENAI_API_KEY=...            # 값은 절대 출력하지 않는다
#   ./run_practice_evalset.sh <결과폴더> [--index <경로>] [그 밖의 run_eval 인자...]
#
# 결과: <결과폴더>/details.jsonl, responses.jsonl, summary.json, api_cost_summary.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?결과 폴더를 첫 인자로 주세요}"
shift || true

EVALSET="${PRACTICE_EVALSET:-${RAG_ROOT:-/srv/rfp}/evalset/practice_items.jsonl}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY 환경변수가 없습니다. (키 값은 출력하지 않습니다)" >&2
  exit 1
fi
echo "✅ OPENAI_API_KEY 존재 확인(값은 출력하지 않음)"

if [[ ! -f "${EVALSET}" ]]; then
  echo "❌ 연습 평가셋을 찾지 못했습니다: ${EVALSET}" >&2
  exit 1
fi
echo "평가셋: ${EVALSET} ($(grep -c . "${EVALSET}")문항)"

python3 "${HERE}/run_eval.py" \
  --evalset "${EVALSET}" \
  --out "${OUT}" \
  --max-item-retries 1 \
  "$@"

echo
echo "=== API 비용 요약 ==="
python3 - "${OUT}/api_cost_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"  생성 요청 {d['total_requests']['generation']}건 / "
      f"임베딩 요청 {d['total_requests']['embedding']}건")
t = d["tokens"]
print(f"  생성 입력 {t['generation_input_tokens']} (캐시 {t['generation_cached_input_tokens']}) / "
      f"출력 {t['generation_output_tokens']} / 임베딩 {t['embedding_tokens']}")
print(f"  총비용 ${d['total_cost_usd']} / 총 실행시간 {d['total_elapsed_ms']}ms")
PY
