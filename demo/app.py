"""
RAG - 데모용 프로토타입 UI
라이브러리: Gradio
구조: src/scripts/answer_pipeline.py 의
    build_runtime()/answer()/answer_to_response() 사용
    1. 첫 시작(build_runtime 1회) -> 2. 대화 세션 유지 -> 3. 화면 표시
실행: python3 demo/app.py
"""

# pip install gradio openai
from __future__ import annotations
import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
import gradio as gr

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
    Stage1Planner,
    Stage2Agent,
)

# 기능 추가 - 고객 회사 정보 및 추천 공고
sys.path.insert(0, str(_ROOT / "tools" / "company_match"))
sys.path.insert(0, str(_ROOT / "src" / "rag"))

import company_match as C
from identity_metadata import reference_datetime_from_config

# 데모 테마 커스텀
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

/* 9) 근거 (Sources) */
.hide-container .md h3,
.hide-container .md p {
    color: #F7F5EF !important;
}

/* 10-1) 기능 선택 — 선택된 탭 글씨·밑줄을 버터색으로  */
button.selected,
.tab-nav button.selected,
.tabs button.selected {
    color: #F4D188 !important;
    border-bottom-color: #F4D188 !important;
}
.tab-nav button,
.tabs button {
    color: #D6DEC8 !important;   /* 비선택 탭 = 연한 글씨*/
}

/* 10-2) 탭 밑줄(선택 표시) — border 외 다른 방식으로 그려지는 경우까지 덮는다 */
.tab-nav button.selected::after,
.tabs button.selected::after,
button.selected::after {
    background: #F4D188 !important;
    background-color: #F4D188 !important;
    border-color: #F4D188 !important;
}
.tab-nav > button.selected,
.tabs > button.selected {
    border-bottom: 2px solid #F4D188 !important;
    box-shadow: inset 0 -2px 0 0 #F4D188 !important;
}

/* 10-3) 탭 밑줄 — 어떤 방식으로 그려지든 덮도록 넓게 (border/box-shadow/가상요소) */
[class*="tab"] button.selected,
[class*="tab"] button[aria-selected="true"] {
    color: #F4D188 !important;
    border-bottom-color: #F4D188 !important;
    box-shadow: inset 0 -3px 0 0 #F4D188 !important;
}
[class*="tab"] button.selected::after,
[class*="tab"] button[aria-selected="true"]::after,
[class*="tab"] button.selected::before,
[class*="tab"] button[aria-selected="true"]::before {
    background: #F4D188 !important;
    background-color: #F4D188 !important;
    border-color: #F4D188 !important;
}

/* 10-4) 오른쪽 결과 영역(진녹 배경 위 마크다운) — 밝은 글씨 */
#cm-priority, #cm-priority *,
#cm-review, #cm-review *,
#cm-urgent, #cm-urgent * {
    color: #F7F5EF !important;
}
#cm-priority h3, #cm-review h3, #cm-urgent h3 {
    color: #F4D188 !important;   /* 결과 소제목은 버터색으로 강조 */
}

/* 11) 여백 */
.gradio-container {
    padding: 28px !important;
}

/* 12-1) 본문 폰트 */
.gradio-container, .gradio-container * {
    font-family: 'Gyeonggi', sans-serif !important;
    font-weight: 300;
}

/* 12-2) 폰트 미적용 요소(폼 라벨·버튼·라디오 등)까지 강제 */
.gradio-container label,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container span,
.gradio-container p,
.gradio-container div {
    font-family: 'Gyeonggi', sans-serif !important;
}
"""


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

    sa = resp.get("structured_answer")
    # 목록형 필드(참가 자격·필수 제출 서류·평가 배점·과업 범위·컨소시엄 요건 등)를
    # 단건 조회하면 계약상 structured_answer가 dict가 아니라 배열 그대로
    # 포장은 단일 필드 dict({field: {status, ...}}) 모양에서만 의미가 있으므로,
    # 그 밖의 모양(list·빈 값 등)은 원본 answer 텍스트를 그대로 보여준다.
    if not isinstance(sa, dict) or not sa:
        return raw
    field_name, detail = next(iter(sa.items()))
    if not isinstance(detail, dict):
        return raw
    status = detail.get("status")

    if status == "value_present":
        value = detail.get("answer_normalized") or detail.get("answer_raw") or raw
        return f"{field_name}: {value} 입니다"
    elif status == "field_absent":
        return f"해당 문서에는 '{field_name}' 항목이 명시되어 있지 않습니다. *자세한 사항은 원문 참조"
    else:
        return raw


# 회사매칭: 회사 입력 정보 + 공고 마감일 임박, print만 Markdown 문자열로 대체
def _doc_label(document_id: str, identity) -> str:
    record = identity.get(document_id) if identity is not None else None
    if record is None or not record.project_name:
        return document_id
    return f"{document_id} · {record.project_name}"


def _location(location) -> str:
    if isinstance(location, str):
        return location
    if not isinstance(location, str):
        return ""
    heading = str(location.get("heading") or "").strip()
    line = location.get("line_start") or location.get("line")
    if heading and line:
        return f"{heading} (line{line})"


def _candidate(item: dict, identity) -> str:
    """후보 1건을 Markdown 불릿으로, (원본 _print_candidate와 로직 동일)"""
    lines = [f"- **{_doc_label(item['document_id'], identity)}**"]
    for field_name in C.MATCHED_FIELDS:
        detail = item["reasons"][field_name]
        if detail["status"] not in {
            C.FIELD_CONFIRMED,
            C.FIELD_REVIEW,
            C.FIELD_CONTRADICTION,
        }:
            continue
        lines.append(f"- {field_name}: {detail['status']} - {detail['reason']}")
        if detail["status"] == C.FIELD_REVIEW and detail["evidence"]:
            lines.append(f"- 원문 근거: {detail['evidence'][:200]}")
            location = _location(detail["location"])
            if location:
                lines.append(f"- 위치: {location}")
    return "\n".join(lines)


def _recommendations(
    company_name: str, results: list[dict], identity, top_n: int = 5
) -> tuple[str, str]:
    """(우선 검토 후보) | (사람 확인 필요 후보) 반환"""
    priority = [i for i in results if i["verdict"] == C.VERDICT_PRIORITY]
    review = [i for i in results if i["verdict"] == C.VERDICT_REVIEW]
    excluded = [i for i in results if i["verdict"] == C.VERDICT_EXCLUDED]

    head = [f"### {company_name} · 우선 검토 후보 {min(len(priority), top_n)}건"]
    if not priority:
        head.append("명시적 조건 일치한 후보 없음")
    else:
        head.extend(_candidate(item, identity) for item in priority[:top_n])
    priority_md = "\n".join(head)

    body = [f"### 육안 확인이 필요 {min(len(review), top_n)}건"]
    if not review:
        body.append("_육안 확인 후보 없음")
    else:
        body.extend(_candidate(item, identity) for item in review[:top_n])
    body.append("")
    body.append(f"명시적 조건 불일치 -> 제외: {len(excluded)}건")
    body.append("* 추천도 실수 할 수 있음. 육안 확인 팔요")
    review_md = "\n".join(body)

    return priority_md, review_md


def _urgent(
    results: list[dict], identity, cfg: dict, top_n: int = 5, urgent_days: int = 14
) -> str:
    """후보('제외'는 제외) 중 마감 임박 공고. (원본 print_urgent_candidates와 로직 동일)"""
    if identity is None:
        return "_마감 정보를 확인할 수 없습니다._"
    candidate_ids = {
        item["document_id"] for item in results if item["verdict"] != C.VERDICT_EXCLUDED
    }
    reference = reference_datetime_from_config(cfg)
    end = reference + timedelta(days=urgent_days)
    urgent = sorted(
        (
            (document_id, record.bid_deadline)
            for document_id, record in identity.records.items()
            if document_id in candidate_ids
            and record.bid_deadline is not None
            and reference <= record.bid_deadline <= end
        ),
        key=lambda item: (item[1], item[0]),
    )[:top_n]
    lines = [f"### 후보 중 {urgent_days}일 안에 마감하는 공고 {len(urgent)}건"]
    if not urgent:
        lines.append("_해당 공고가 없습니다._")
    else:
        lines.extend(
            f"- {_doc_label(document_id, identity)} · {deadline:%Y-%m-%d}"
            for document_id, deadline in urgent
        )
    return "\n".join(lines)


def on_match(
    company_name: str,
    region: str,
    fields_str: str,
    certs_str: str,
    consortium_choice: str,
):
    """폼(회사 정보) 제출 -> 프로필 구성 -> 매칭 -> 3개의 md(우선 검토 후보, 육안 확인 후보, 마감 임박) 반환"""
    if not company_name.strip() or not region.strip():
        return "회사 이름과 지역 설정은 필수입니다.", "", ""

    profile = C.CompanyProfile(
        company_name=company_name.strip(),
        region=region.strip(),
        business_fields=[s.strip() for s in fields_str.split(".") if s.strip()],
        certifications=[s.strip() for s in certs_str.split(".") if s.strip()],
        consortium_needed={"예": True, "아니오": False, "모름": None}.get(
            consortium_choice
        ),
    )
    eligible_ids = (
        set(_RT["registry_scope"].eligible_ids)
        if _RT["registry_scope"] is not None
        else None
    )
    try:
        results = C.match_company(profile, _RT["table"], eligible_ids)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"매칭 중 오류가 발생했습니다.: {type(e).__name__}", "", ""

    priority_md, review_md = _recommendations(
        company_name.strip(), results, _RT["identity"]
    )
    urgent_md = _urgent(results, _RT["identity"], _CFG)
    return priority_md, review_md, urgent_md


_STAGE_CACHE: dict = {}


def _stage1_planner() -> Stage1Planner:
    eligible = (
        _RT["registry_scope"].eligible_ids
        if _RT["registry_scope"] is not None
        else None
    )
    if "planner" not in _STAGE_CACHE:
        _STAGE_CACHE["planner"] = Stage1Planner(_CFG, _RT["identity"], eligible)
    _STAGE_CACHE["planner"].reset_usage()
    return _STAGE_CACHE["planner"]


def _stage2_agent() -> Stage2Agent:
    eligible = (
        _RT["registry_scope"].eligible_ids
        if _RT["registry_scope"] is not None
        else None
    )
    if "agent2" not in _STAGE_CACHE:
        _STAGE_CACHE["agent2"] = Stage2Agent(_CFG, _RT["identity"], eligible)
    _STAGE_CACHE["agent2"].reset_usage()
    return _STAGE_CACHE["agent2"]


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
            get_stage1_planner=_stage1_planner,
            get_stage2_agent=_stage2_agent,
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

        with gr.Tabs():
            with gr.Tab("회사 매칭"):
                gr.Markdown("회사 정보를 입력하면 알맞은 입찰 공고를 추천합니다.")
                with gr.Row():
                    with gr.Column(scale=2):
                        cm_name = gr.Textbox(label="회사 이름")
                        cm_region = gr.Textbox(label="지역")
                        cm_fields = gr.Textbox(
                            label="사업분야",
                            placeholder="에: 소프트웨어, 시스템구축",
                        )
                        cm_certs = gr.Textbox(
                            label="보유 자격·신고증",
                            placeholder="에: 정보통신공사업",
                        )
                        cm_consortium = gr.Radio(
                            ["예", "아니오", "모름"],
                            value="모름",
                            label="공동수급 여부",
                        )
                        cm_btn = gr.Button("공고 매칭", variant="primary")
                    with gr.Column(scale=3):
                        cm_priority = gr.Markdown(
                            "_우선 검토 후보_",
                            elem_id="cm-priority",
                        )
                        cm_review = gr.Markdown(
                            "_육안 확인 필요 후보_",
                            elem_id="cm-review",
                        )
                        cm_urgent = gr.Markdown("_마감 임박 공고_", elem_id="cm-urgent")

            with gr.Tab("질문하기"):
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

        cm_btn.click(
            on_match,
            inputs=[cm_name, cm_region, cm_fields, cm_certs, cm_consortium],
            outputs=[cm_priority, cm_review, cm_urgent],
        )

        return demo


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경변수 없음", file=sys.stderr)

    ui = build_ui()
    _ASSETS = str(Path(__file__).resolve().parent / "assets")
    ui.launch(share=False, allowed_paths=[_ASSETS], css=_CUSTOM_CSS)
