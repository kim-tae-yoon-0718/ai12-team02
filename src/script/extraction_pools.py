"""
추출형 15개 문항 재료 분석

1. selection_pools.json에서 필드 상태별 후보 문서를 가져와
   추출형 하위유형(본문값/표값/항목없음/목록값)에 문서 1건씩 배정
2. 확정된 (문서ID, line) 쌍으로 규칙{document, section, ref_no, line}에 맞는 location 딕셔너리 조회

주의: 1번 배정 결과는 "왜 이 문서를 후보로 골랐는가"의 기록일 뿐,
실제 answer_raw·location은 원문(chunks_v3) 직접 대조로 별도 확정
그 과정에서 EXT-04(RFP-000004→019), EXT-12(RFP-000013→030)는
chunking 마크다운 파싱 버그로 문서 자체가 교체
"""
import json

RAG_ROOT = "/srv/rfp"
CHUNK_PATH = f"{RAG_ROOT}/shared_data/processed/chunks_v3/chunks.jsonl"
POOLS_PATH = f"{RAG_ROOT}/evalset/v1/_work/selection_pools.json"

with open(POOLS_PATH, encoding="utf-8") as f:
    pools = json.load(f)

# --- 1. 하위유형별 후보 문서 1건씩 배정 (중복 없이) ---
# EXT-07(공고 마감일)은 CSV 메타데이터 출처라 제외
assignments = {
    "EXT-01 본문값 참가자격(critical)":     "single_present__참가 자격(면허·실적)",
    "EXT-02 본문값 지역제한(critical)":     "single_present__지역제한",
    "EXT-03 본문값 컨소시엄요건(critical)":  "single_present__컨소시엄 요건",
    "EXT-04 본문값 사업기간(major)":        "single_present__사업기간",
    "EXT-05 표값 예산(major)":             "single_present__예산",
    "EXT-06 표값 평가배점(major)":          "single_present__평가 배점",
    "EXT-08 표값 제출방식(critical)":       "single_present__제출 방식",
    "EXT-09 항목없음 지역제한(critical)":    "negative_absent__지역제한",
    "EXT-10 항목없음 컨소시엄요건(critical)": "negative_absent__컨소시엄 요건",
    "EXT-11 항목없음 필수제출서류(major)":    "negative_absent__필수 제출 서류",
    "EXT-12 목록값 필수제출서류(major)":      "single_present__필수 제출 서류",
    "EXT-13 목록값 과업범위(minor)":         "single_present__과업 범위",
}

used_docs = set()
for label, pool_key in assignments.items():
    candidates = [d for d in pools[pool_key] if d not in used_docs]
    pick = candidates[0] if candidates else None
    if pick:
        used_docs.add(pick)
    print(f"{label:40s} -> {pick} (후보 {len(candidates)}건)")


# --- 2. 확정된 (document_id, line)으로 정확한 location 조회 ---
def load_chunks_by_key(path):
    by_key = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by_key[(c["document_id"], c.get("md_line_start"))] = c
    return by_key


def get_location(by_key, document_id, line_start):
    c = by_key.get((document_id, line_start))
    if not c:
        return None
    section_path = c.get("section_path")
    return {
        "document": c["document_id"],
        "section": section_path[-1] if section_path else None,
        "ref_no": c.get("location_label"),
        "line": c.get("md_line_start"),
    }


if __name__ == "__main__":
    by_key = load_chunks_by_key(CHUNK_PATH)

    # 최종 확정된 문서×line (문서 교체분 포함, EXT-07/09/10/11/14/15 제외)
    confirmed = [
        ("EXT-01", "RFP-000002", 1445),
        ("EXT-02", "RFP-000005", 693),
        ("EXT-03", "RFP-000003", 1384),
        ("EXT-04", "RFP-000019", 79),
        ("EXT-05", "RFP-000007", 103),
        ("EXT-06", "RFP-000008", 675),
        ("EXT-08", "RFP-000010", 1042),
        ("EXT-12", "RFP-000030", 1098),
        ("EXT-13", "RFP-000012", 104),
    ]

    print("\n=== location 재확인 ===")
    for label, doc_id, line in confirmed:
        loc = get_location(by_key, doc_id, line)
        print(label, "->", json.dumps(loc, ensure_ascii=False))