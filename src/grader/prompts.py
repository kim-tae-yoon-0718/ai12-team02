from __future__ import annotations

import re
from pathlib import Path


_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptRepository:
    """
    채점 프롬프트 파일 로더.

    ★ 실제 팀 확정 프롬프트(judge_*.v1.md)는 `{{변수명}}` 치환 방식을 쓴다.
      이 저장소는 프롬프트 파일을 임의로 재작성하지 않으므로, 렌더러가 그 문법에
      맞춰야 한다 — 반대로 렌더러 편의를 위해 프롬프트 문법을 바꾸지 않는다.

    prompt_files 는 {judge_name: 경로} 매핑이다. 파일명은 prompts/<name>.<version>.md
    규약(【23】)을 따른다 — 버전이 바뀌면 이전 점수와의 비교가 무효가 될 수 있으므로
    파일명 자체에 버전을 남긴다.
    """

    def __init__(self, prompt_files: dict[str, Path]) -> None:
        self.prompt_files = prompt_files

    def version_of(self, judge_name: str) -> str:
        path = self.prompt_files[judge_name]
        stem = Path(path).stem  # "judge_faithfulness.v1" -> stem 은 "judge_faithfulness.v1"
        if "." in stem:
            return stem.rsplit(".", 1)[-1]
        return "unknown"

    def render(self, judge_name: str, variables: dict[str, object]) -> str:
        path = self.prompt_files[judge_name]
        if not Path(path).exists():
            raise FileNotFoundError(f"Judge prompt 파일이 없습니다: {path}")
        text = Path(path).read_text(encoding="utf-8")

        def _sub(m: re.Match) -> str:
            key = m.group(1)
            if key in variables:
                return str(variables[key])
            return m.group(0)  # safe-substitute: 값이 없으면 그대로 둔다

        return _VAR.sub(_sub, text)
