#!/usr/bin/env bash
# 2단계 검증 — 실제 OpenAI API로 질문 1건만 돌려본다(비용 최소).
#
# 사용법:
#   source /srv/rfp/venv/bin/activate
#   export RAG_ROOT=/srv/rfp
#   export OPENAI_API_KEY=...            # 값은 절대 출력하지 않는다
#   ./real_api_smoke_test.sh [질문] [--index <경로>]
#
# 무료 테스트(pytest)가 모두 통과한 뒤에만 실행한다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUESTION="${1:-RFP-000038의 예산이 얼마인지 근거와 함께 설명해줘}"
shift || true

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY 환경변수가 없습니다. (키 값은 출력하지 않습니다)" >&2
  exit 1
fi
echo "✅ OPENAI_API_KEY 존재 확인(값은 출력하지 않음)"

if [[ -z "${RAG_ROOT:-}" ]]; then
  echo "❌ RAG_ROOT 환경변수가 없습니다. export RAG_ROOT=/srv/rfp" >&2
  exit 1
fi

echo "질문: ${QUESTION}"
python3 "${HERE}/answer_pipeline.py" --question "${QUESTION}" "$@"
