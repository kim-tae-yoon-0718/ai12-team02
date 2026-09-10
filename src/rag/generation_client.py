"""
J — 답변 생성. 오픈AI 트랙, generation_model=gpt-5-mini(base.yaml 확정).

프롬프트 지시 4종을 항상 포함한다: ① 근거에만 기반 ② 근거에 없으면 모른다
③ 출처 표기 ④ 형식·톤·길이. 프롬프트 본문은 코드에 넣지 않고
prompts/generate_v1.txt에서 읽는다(message.txt 4번 확정).

[2026-09-02 — GPT-5 Mini 요청 인자 정리]
gpt-5 계열은 다음 인자를 지원하지 않는다. 요청에 **아예 싣지 않는다**:
  - temperature (기본값만 허용)
  - top_p
  - logprobs
  - max_tokens  → 대신 max_completion_tokens 사용
base.yaml은 `temperature: null`(= 보내지 않음)로 명시하고,
`max_completion_tokens: 4096`을 시작값으로 둔다.
⚠️ GPT-5 Mini에서는 API 기본값을 사용하며 동일 문장의 완전한 재현은 보장하지 않음.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pricing import Usage

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# 이 접두어로 시작하는 모델은 temperature/top_p/logprobs/max_tokens를 못 받는다
_RESTRICTED_PARAM_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def model_rejects_sampling_params(model: str) -> bool:
    return any(model.startswith(p) for p in _RESTRICTED_PARAM_MODEL_PREFIXES)


def _find_prompt_file(name: str) -> Path:
    override = os.environ.get("RAG_PROMPT_DIR")
    if override:
        p = Path(override) / name
        if p.exists():
            return p
        raise RuntimeError(f"RAG_PROMPT_DIR이 가리키는 파일이 없습니다: {p}")

    here = Path(__file__).resolve().parent
    for candidate_root in [here, *here.parents]:
        candidate = candidate_root / "prompts" / name
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"prompts/{name}을 찾지 못했습니다. RAG_PROMPT_DIR 환경변수로 직접 지정하거나 "
        f"prompts/{name}을 만들어 두세요. 프롬프트를 코드에 하드코딩하지 않습니다."
    )


def _load_system_prompt_template(name: str | None = None) -> str:
    """프롬프트 파일 이름은 base.yaml의 prompt_generate 에서 온다(코드에 고정하지 않음).
    ⚠️ 팀 규약: 프롬프트를 고칠 땐 같은 파일을 수정하지 말고 _v2 를 새로 만든다.

    ⚠️ 2026-09-03: 예전에는 이름이 비면 조용히 "generate_v2.txt"로 되돌아갔다.
       그러면 실험 config가 `prompt_generate: null`로 덮어쓰거나 cfg를 손으로 만들었을 때
       **아무 신호 없이 옛 프롬프트로 실행**되고, 산출물만 봐서는 알 수 없다.
       (실제로 generate_v3.txt로 막으려던 '근거 밖 항목 추가'가 그 실행에서만 되살아난다.)
       코드가 조용히 기본값을 고르지 않는다는 팀 원칙대로, 이제는 즉시 중단한다.
    """
    if not name:
        raise RuntimeError(
            "생성 프롬프트 파일 이름(prompt_generate)이 비어 있습니다. "
            "base.yaml 또는 실험 config에서 값을 채운 뒤 다시 실행하세요 — "
            "코드가 조용히 옛 프롬프트로 되돌아가지 않습니다."
        )
    return _find_prompt_file(name).read_text(encoding="utf-8")


# 근거 인용 규약 (결함 1-4) — 모델이 "실제로 쓴" 근거만 돌려주게 한다.
EVIDENCE_ID_RE = re.compile(r"\bE\s*(\d+)\b", re.IGNORECASE)
# ⚠️ 2026-09-02 수정: 예전엔 줄 **처음**에 오는 표식만 인식했다(`^...` + MULTILINE).
#    실제 gpt-5-mini 는 "(근거: …) USED_EVIDENCE: E5" 처럼 문장 **중간**에, 그것도
#    여러 번 내보냈다 — 그 결과 표식을 못 찾아 인용이 0건이 되고 표식 문자열이
#    답변 본문에 그대로 새어 나왔다. 위치와 횟수에 상관없이 전부 걷어낸다.
# 표식 + 번호 목록(또는 NONE)까지만 소비한다 — 뒤에 이어지는 문장을 삼키지 않는다.
USED_EVIDENCE_RE = re.compile(
    r"[ \t>*\-]*\(?USED[ _]?EVIDENCE\s*[:：]\s*"
    r"(?P<ids>NONE|없음|E\s*\d+(?:\s*[,、·/]\s*(?:E\s*)?\d+)*)?\)?[ \t]*",
    re.IGNORECASE,
)


def split_used_evidence(text: str) -> tuple[str, list[int] | None]:
    """생성 결과에서 USED_EVIDENCE 표식을 전부 떼어낸다.

    반환: (본문, 사용한 근거 번호 목록).
      - 표식이 하나도 없으면 두 번째 값이 None (= 규약 미준수 → 인용 안 붙임)
      - 표식이 있고 번호가 하나도 없으면(전부 NONE) 빈 목록
      - 표식이 여러 번 나오면 번호를 **합집합**으로 모은다
    본문에는 표식이 남지 않는다(문장 중간에 있었더라도).
    """
    raw = text or ""
    matches = list(USED_EVIDENCE_RE.finditer(raw))
    if not matches:
        return raw.strip(), None

    ids: set[int] = set()
    for m in matches:
        # 이미 목록으로 읽은 'E1, 2'의 2도 보존한다. 번호 길이를 잘라 E1000을
        # E100으로 바꾸지 않는다. 존재 여부는 실제 컨텍스트 개수로 호출측이 확인한다.
        ids.update(int(number) for number in re.findall(r"\d+", m.group("ids") or ""))

    body = USED_EVIDENCE_RE.sub(" ", raw)   # 앞뒤 단어가 붙지 않게 한 칸 남긴다
    body = re.sub(r"[ \t]+\n", "\n", body)      # 표식 제거로 생긴 줄 끝 공백
    body = re.sub(r"\n{3,}", "\n\n", body)       # 빈 줄 과다 정리
    body = re.sub(r"[ \t]{2,}", " ", body)
    return body.strip(), sorted(ids)


class GenerationResponseError(RuntimeError):
    """생성이 정상 완료되지 않았거나 실제 답변 본문이 없는 경우."""


def build_request_kwargs(
    cfg: dict[str, Any], model: str, messages: list[dict],
) -> dict[str, Any]:
    """실제로 API에 보낼 인자를 만든다(테스트가 실호출 없이 검증할 수 있게 분리).

    - 지원 안 하는 인자는 키 자체를 넣지 않는다(값이 None인 채로도 안 보낸다).
    - max_tokens는 어떤 경우에도 넣지 않는다 — max_completion_tokens만 쓴다.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    max_completion_tokens = cfg.get("max_completion_tokens")
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_completion_tokens)

    if not model_rejects_sampling_params(model):
        # temperature/top_p/logprobs는 "설정에 값이 있을 때만" 싣는다.
        # null은 "보내지 않음"이라는 뜻이다(예전처럼 0을 확정값처럼 적어놓고
        # 실제로는 무시하는 모순을 만들지 않는다).
        if cfg.get("temperature") is not None:
            kwargs["temperature"] = cfg["temperature"]
        if cfg.get("top_p") is not None:
            kwargs["top_p"] = cfg["top_p"]
        if cfg.get("logprobs") is not None:
            kwargs["logprobs"] = cfg["logprobs"]
    return kwargs


class GenerationClient:
    def __init__(self, cfg: dict[str, Any]):
        if cfg.get("generation_provider") != "openai":
            raise NotImplementedError("이 클라이언트는 오픈AI 트랙 전용입니다.")
        if not cfg.get("data_egress_confirmed", False):
            raise RuntimeError(
                "data_egress_confirmed=false — 데이터 반출 확인이 안 끝났습니다."
            )
        if OpenAI is None:
            raise ImportError("pip install openai --break-system-packages")
        if not os.environ.get("OPENAI_API_KEY"):
            # 값을 읽어 보관하지 않는다 — 존재 여부만 확인하고 SDK가 직접 읽게 둔다
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
        self.client = OpenAI()
        model = cfg.get("generation_model")
        if not model:
            raise RuntimeError(
                "generation_model이 비어 있습니다. 코드가 조용히 기본값을 고르지 않습니다."
            )
        self.cfg = cfg
        self.model = model
        self.max_completion_tokens = cfg.get("max_completion_tokens")
        # 실제로 어떤 프롬프트 파일이 실렸는지 기록해 둔다(산출물에 남겨 관측 가능하게).
        self.prompt_file = cfg.get("prompt_generate")
        self._system_prompt_template = _load_system_prompt_template(self.prompt_file)
        # 문항 단위 사용량 — run_eval이 문항 시작 때마다 reset_usage()로 초기화한다
        self.usage = Usage()
        self.last_request_kwargs: dict[str, Any] | None = None
        # 직전 generate() 에서 모델이 "실제로 썼다"고 밝힌 근거 번호(결함 1-4).
        # None = 규약 미준수(인용 붙이지 않음), [] = 원문 근거 미사용.
        self.last_used_evidence: list[int] | None = None

    def reset_usage(self) -> None:
        """문항 시작 시 호출 — 이전 문항 사용량이 다음 문항으로 복사되지 않게."""
        self.usage = Usage()

    def build_messages(
        self, question: str, context_chunks: list[str],
        structured_context: str | None = None,
        format_instruction: str = "간결하고 명확하게 답하세요.",
    ) -> list[dict]:
        system = self._system_prompt_template.format(format_instruction=format_instruction)
        parts: list[str] = []
        if structured_context:
            parts.append(
                "[공식 확정 값 — 구조화 자료]\n"
                "아래 값은 팀이 검증해 확정한 공식 구조화 자료입니다. "
                "원문 청크의 주변 문장을 근거로 이 값을 바꾸거나 다시 계산하지 마세요.\n"
                + structured_context
            )
        if context_chunks:
            numbered = [f"[E{i}] {c}" for i, c in enumerate(context_chunks, 1)]
            context = "\n\n---\n\n".join(numbered)
        else:
            context = "(원문 근거 없음)"
        parts.append("[원문 근거 — 설명·문맥 보완용. 각 근거의 번호(E1, E2…)를 "
                     "USED_EVIDENCE 줄에 쓰세요]\n" + context)
        parts.append(f"[질문]\n{question}")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        format_instruction: str = "간결하고 명확하게 답하세요.",
        structured_context: str | None = None,
    ) -> str:
        # 이전 요청의 근거 번호를 다음 요청 실패 시 재사용하지 않도록 먼저 비운다.
        self.last_used_evidence = None
        messages = self.build_messages(
            question, context_chunks,
            structured_context=structured_context,
            format_instruction=format_instruction,
        )
        kwargs = build_request_kwargs(self.cfg, self.model, messages)
        self.last_request_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}

        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            cached = 0
            details = getattr(resp.usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            self.usage.add_generation(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                cached_tokens=cached,
            )
        if not resp.choices:
            raise GenerationResponseError("생성 응답에 답변 후보가 없습니다.")
        choice = resp.choices[0]
        # 실제 SDK는 종료 사유를 제공한다. 옛 시험 대역에만 필드가 없을 수 있다.
        finish_reason = getattr(choice, "finish_reason", "stop")
        if finish_reason != "stop":
            raise GenerationResponseError(
                f"생성이 정상 종료되지 않았습니다(finish_reason={finish_reason}). "
                "중간에 잘린 답을 완성된 답변으로 사용하지 않습니다."
            )
        if getattr(choice.message, "refusal", None):
            raise GenerationResponseError("생성 API가 요청을 거부하여 답변을 완료하지 못했습니다.")
        raw = choice.message.content or ""
        if not isinstance(raw, str):
            raise GenerationResponseError("생성 응답의 본문이 텍스트가 아닙니다.")
        body, used = split_used_evidence(raw)
        if not body:
            raise GenerationResponseError("생성 응답에 답변 본문이 없습니다(빈 출력 또는 근거 표식만 있음).")
        self.last_used_evidence = used
        return body
