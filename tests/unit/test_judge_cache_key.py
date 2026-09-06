"""결함 B — 심판 결과 캐시 오염 (2026-09-04).

★예전 키에는 질문·답변·컨텍스트·코퍼스·추출표만 들어 있었다. 그래서 심판 모델을
  바꾸거나 채점 프롬프트를 고쳐도 **예전 판정을 그대로 다시 썼다.**
  "심판을 바꿨는데 점수가 안 변한다"가 그 증상이다.
"""
import inspect

import pytest

from grader.cache import CACHE_KEY_SCHEMA, build_cache_key

BASE = dict(
    evaluation_id="QA-001", judge_name="judge_faithfulness",
    question="왜 하는 사업인가?", answer="대중교통 서비스 개선을 위해서다.",
    contexts=[{"document": "RFP-000001", "text": "…"}],
    corpus_version="v2", extraction_table_version="v3",
    judge_model="claude-sonnet-5", judge_family="anthropic",
    judge_provider="openai_compatible", judge_temperature=0.0, judge_tier="final",
    prompt_version="v1", prompt_sha256="a" * 64,
    scorer_version="v2", evalset_version="v1-candidate.3",
)


def key(**over):
    return build_cache_key(**{**BASE, **over})


def test_same_inputs_and_settings_give_the_same_key():
    assert key() == key()


@pytest.mark.parametrize("field,value", [
    ("judge_model", "gpt-5-mini"),
    ("judge_provider", "anthropic_native"),
    ("judge_temperature", 0.7),
    ("judge_tier", "dev"),
    ("prompt_version", "v2"),
    ("prompt_sha256", "b" * 64),
    ("scorer_version", "v3"),
    ("evalset_version", "v1-candidate.2"),
    ("judge_name", "judge_checkpoint"),
    ("corpus_version", "v1"),
    ("extraction_table_version", "v2"),
])
def test_changing_any_judging_input_changes_the_key(field, value):
    assert key(**{field: value}) != key(), f"{field} 를 바꿨는데 캐시 키가 같다"


def test_prompt_content_change_alone_changes_the_key():
    """★버전 문자열은 그대로 두고 프롬프트 내용만 고치는 일이 흔하다.
    버전만 보면 그 변경을 놓쳐 예전 판정을 재사용한다."""
    assert key(prompt_sha256="c" * 64) != key(prompt_version="v1")


def test_family_aliases_do_not_split_the_cache():
    """'gpt' 와 'openai' 가 다른 키를 만들면 같은 심판이 캐시를 두 벌 갖는다."""
    assert key(judge_family="openai") == key(judge_family="gpt")
    assert key(judge_family="anthropic") == key(judge_family="claude")
    assert key(judge_family="openai") != key(judge_family="anthropic")


def test_old_format_cache_cannot_be_reused():
    """예전 형식(심판 설정 없음)으로 만든 키는 새 코드에서 절대 나오지 않는다."""
    import hashlib
    import json
    old_payload = {
        "evaluation_id": BASE["evaluation_id"], "judge_name": BASE["judge_name"],
        "question": BASE["question"], "answer": BASE["answer"],
        "contexts": BASE["contexts"], "corpus_version": BASE["corpus_version"],
        "extraction_table_version": BASE["extraction_table_version"],
    }
    old_key = hashlib.sha256(json.dumps(
        old_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    assert key() != old_key
    assert CACHE_KEY_SCHEMA == "v2"


def test_new_fields_are_required_not_silently_defaulted():
    """★기본값이 있으면 호출부가 빠뜨려도 조용히 예전 동작으로 돌아간다."""
    sig = inspect.signature(build_cache_key)
    for name in ("judge_model", "judge_family", "judge_provider", "judge_temperature",
                 "judge_tier", "prompt_version", "prompt_sha256",
                 "scorer_version", "evalset_version"):
        assert name in sig.parameters, f"{name} 이 캐시 키에 없다"
        assert sig.parameters[name].default is inspect.Parameter.empty, \
            f"{name} 에 기본값이 있으면 호출부가 빠뜨려도 조용히 통과한다"


def test_runner_passes_every_field(tmp_path):
    """호출부가 실제로 모든 축을 넘기는지 — 서명만 맞고 안 넘기면 소용없다."""
    import grader.runner as R
    src = inspect.getsource(R.GraderRunner.run_one)
    for name in ("judge_model=", "judge_family=", "judge_provider=",
                 "judge_temperature=", "judge_tier=", "prompt_version=",
                 "prompt_sha256=", "scorer_version=", "evalset_version="):
        assert name in src, f"run_one 이 {name} 를 캐시 키에 안 넘긴다"
