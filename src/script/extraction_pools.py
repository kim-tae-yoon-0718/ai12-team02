import json

"""
추출형 문항 배분을 위한, 데이터 밀집도 확인
"""
with open('/srv/rfp/evalset/v1/_work/selection_pools.json', encoding='utf-8') as f:
    extrat = json.load(f)

assignments = {
    "EXT-01 본문값 참가자격(critical)": "single_present__참가 자격(면허·실적)",
    "EXT-02 본문값 지역제한(critical)": "single_present__지역제한",
    "EXT-03 본문값 컨소시엄요건(critical)": "single_present__컨소시엄 요건",
    "EXT-04 본문값 사업기간(major)": "single_present__사업기간",
    "EXT-05 표값 예산(major)": "single_present__예산",
    "EXT-06 표값 평가배점(major)": "single_present__평가 배점",
    "EXT-07 표값 공고일(major)": "single_present__공고일",
    "EXT-08 표값 제출방식(critical)": "single_present__제출 방식",
    "EXT-09 항목없음 지역제한(critical)": "negative_absent__지역제한",
    "EXT-10 항목없음 컨소시엄요건(critical)": "negative_absent__컨소시엄 요건",
    "EXT-11 항목없음 필수제출서류(major)": "negative_absent__필수 제출 서류",
    "EXT-12 목록값 필수제출서류/selection_to_extraction(major)": "single_present__필수 제출 서류",
    "EXT-13 목록값 과업범위(minor)": "single_present__과업 범위",
}

used = set()
for label, key in assignments.items():
    candidates = [d for d in extrat[key] if d not in used]
    pick = candidates[0] if candidates else None
    if pick:
        used.add(pick)
    print(f"{label:45s} -> {pick} (후보 {len(candidates)}건)")