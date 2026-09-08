"""
RAG - 데모용 프로토타입 UI
라이브러리: Gradio
구조: src/scripts/answer_pipeline.py 의
    build_runtime()/answer()/answer_to_response() 사용
    1. 첫 시작(build_runtime 1회) -> 2. 대화 세션 유지 -> 3. 화면 표시
실행: python demo/app.py
"""

# pip install gradio openai
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import gradio as gr

_CUSTOM_CSS = """
/* 1) 폰트 (경기서체) */
@font-face {
    font-family: 'Gyeonggi';
    src: url('/gradio_api/file=/home/spai1204/ai12-team02/demo/assets/Title_Light.otf') format('opentype');
    font-weight: 300;
}
@font-face {
    font-family: 'Gyeonggi';
    src: url('/gradio_api/file=/home/spai1204/ai12-team02/demo/assets/Title_Medium.otf') format('opentype');
    font-weight: 500;
}

/* 2) 배경 */
gradio-app {
    background: #657652 !important;
    background-color: #657652 !important;
}
html, body, .gradio-container, .gradio-container .main, .app {
    background: #657652 !important;
    background-color: #657652 !important;
}

/* 3) 팔레트 */
.gradio-container {
    --body-text-color: #2E2A24;            
    --background-fill-primary: #F7F5EF;    
    --background-fill-secondary: #EEF2E8; 
    --border-color-primary: #DDE3D4;
    --input-border-color: #C7CFBA;
    --input-text-color: #2E2A24;
    --input-placeholder-color: #8A9179;

    --button-primary-background-fill: #55643F;     
    --button-primary-text-color: #FFFFFF;
    --button-primary-background-fill-hover: #445133;
    --button-primary-text-color-hover: #FFFFFF;
    --button-secondary-background-fill: #F7F5EF;
    --button-secondary-text-color: #55643F;
    --button-secondary-background-fill-hover: #EAEFE2;
    --button-secondary-text-color-hover: #55643F;
}

/* 4) 카드(챗봇·입력창) */
.block:not(.hide-container) {
    background: #F7F5EF !important;
    border: 1px solid #DDE3D4 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
}

/* 5) 질문 입력창 */
#question-box textarea,
#question-box input {
    background: #FFFFFF !important;
    color: #2E2A24 !important;
}

/* 6) 근거 패널 */
#sources-panel {
    background: #F7F5EF !important;
    border: 1px solid #DDE3D4 !important;
    border-left: 4px solid #55643F !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
}

/* 7) 챗 말풍선 */
.message.bot {
    background: #FFFFFF !important;
    color: #2E2A24 !important;
}
.message.user {
    background: #F4D188 !important;
    color: #2E2A24 !important;
}

/* 8) 제목 */
#app-title h2 {
    font-family: 'Gyeonggi', sans-serif;
    font-weight: 500;
    color: #F7F5EF;
    letter-spacing: -0.01em;
}

/* 근거 (Sources) */
.hide-container .md h3,
.hide-container .md p {
    color: #F7F5EF !important;
}

/* 9) 여백 */
.gradio-container {
    padding: 28px !important;
}

/* 10) 본문 폰트 */
.gradio-container, .gradio-container * {
    font-family: 'Gyeonggi', sans-serif !important;
    font-weight: 300;
}
"""


# src/scripts 에 __init__이 없으므로, 직접 경로 지정
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "scripts"))

from answer_pipeline import (
    build_runtime,
    answer,
    answer_to_response,
    SessionState,
    load_config,
    EmbeddingClient,
    GenerationClient,
)


# runtime (시작 시 1회 로드)
# Gradio는 argparse로 args 생성 못함
# Namespace = None -> build_runtime(_try_derive)가 자동 조립
def _args() -> argparse.Namespace:
    return argparse.Namespace(
        index=None,
        extraction_table=None,
        extraction_metadata=None,
        identity=None,
        registry=None,
        chunks=None,
        experiment_config=None,
        allow_no_chunks=False,
        allow_no_deadline_filter=False,
        allow_no_identity=False,
        allow_unofficial_table=False,
    )


_ARGS = _args()
_CFG = load_config(_ARGS.experiment_config)
_RT = build_runtime(
    _ARGS, _CFG
)  # {"store", "table", "identity", "locator", "registry_scope", ...}
_cache: dict = (
    {}
)  # 클라이언트 팩토리: 세션 동안 클라이언트 1개 사용 (CLI main() 캐시 그대로)


def _embed() -> EmbeddingClient:
    if "embed" not in _cache:
        _cache["embed"] = EmbeddingClient(_CFG)
    _cache["embed"].reset_usage()
    return _cache["embed"]


def _generate() -> GenerationClient:
    if "gen" not in _cache:
        _cache["gen"] = GenerationClient(_CFG)
    _cache["gen"].reset_usage()
    return _cache["gen"]


# answer 표시
def _source(resp: dict) -> str:
    """resp['citations'](구조화된 근거) -> 'document > section > ref_no' Markdown 불릿으로 변환"""
    citations = resp.get("citations") or []
    if not citations:
        return "_근거를 표시할 항목이 없습니다._"
    lines = []
    for c in citations:
        parts = [str(c[k]) for k in ("document", "section", "ref_no") if c.get(k)]
        lines.append("- " + " > ".join(parts) if parts else f"- {c}")
    return "\n".join(lines)


# 답변 말투 수정 (표시 계층 포장 — src·프롬프트 불변, 채점 대상 answer() 출력은 원본 유지)
def _decorate(resp: dict) -> str:
    """추출/거절 route일 때만 답변 말투를 표시용으로 포장"""
    raw = resp.get("answer", "") or "_(빈 응답)_"
    route = resp.get("route", "")
    if route != "추출테이블_값조회":
        return raw

    sa = resp.get("structured_answer") or {}
    if not sa:
        return raw
    field_name, detail = next(iter(sa.items()))
    status = detail.get("status")

    if status == "value_present":
        value = detail.get("answer_normalized") or detail.get("answer_raw") or raw
        return f"{field_name}: {value} 입니다"
    elif status == "field_absent":
        return f"해당 문서에는 '{field_name}' 항목이 명시되어 있지 않습니다. *자세한 사항은 원문 참조"
    else:
        return raw


# 채팅 세션 기억 유지
def respond(message: str, history: list, session: SessionState | None):
    if not message or not message.strip():
        return "질문을 입력해 주세요.", "", session

    if session is None:
        session = SessionState()  # 대화 첫 턴에만 생성
    try:
        result = answer(
            message,
            _RT["store"],
            _embed,
            _generate,
            _RT["table"],
            _CFG,
            identity=_RT["identity"],
            session=session,
            locator=_RT["locator"],
            registry_scope=_RT["registry_scope"],
        )
        resp = answer_to_response(message, result)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"오류 발생: {type(e).__name__}", "", session

    answer_text = _decorate(resp)
    source_md = _source(resp)

    return answer_text, source_md, session


# UI
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="입찰메이트 RAG 데모") as demo:
        gr.Markdown("## 입찰메이트 — RFP 입찰 컨설팅 RAG (데모)", elem_id="app-title")

        session_state = gr.State(None)  # 대화별 SessionState 저장
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(
                    placeholder="예: 오늘 등록된 사업 찾아줘",
                    label="질문",
                    elem_id="question-box",
                )
            with gr.Column(scale=2):
                gr.Markdown("### 근거 (Sources)")
                sources_box = gr.Markdown(
                    "_질문에 대한 근거는 여기 표시됩니다._",
                    elem_id="sources-panel",
                )

        def submit(message, chat_history, session):
            answer_text, sources_md, new_session = respond(
                message, chat_history, session
            )
            chat_history = (chat_history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer_text},
            ]
            return chat_history, sources_md, new_session, ""

        msg.submit(
            submit,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, sources_box, session_state, msg],
        )

        return demo


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경변수 없음", file=sys.stderr)

    ui = build_ui()
    _ASSETS = str(Path(__file__).resolve().parent / "assets")
    ui.launch(share=False, allowed_paths=[_ASSETS], css=_CUSTOM_CSS)
