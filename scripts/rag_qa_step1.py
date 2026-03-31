"""
RAG QA Pipeline — Step 0+1: 필터링 + 구조 분리

입력: nuscale_rai_qa_dataset.json, levy_rai_qa_dataset.json
출력: step1_structured.json

Step 0: 필터링
  - relevant_docs 없는 RAI 제외
  - section_number ↔ 문서 매칭 안 되는 것 제외
  - 섹션당 15개 상한

Step 1: 구조 분리
  - regulatory_background / core_question 분리
  - extracted_refs (GDC, CFR, RG, SRP, DSRS)
  - 난이도 자동 분류 (Easy/Medium/Hard)

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/rag_qa_step1.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── 경로 ─────────────────────────────────────────────
QA_DIR = Path("/home/bbo/RAG_Baseline/qadataset")
META_PATH = Path("/home/bbo/doc_search/data/doc_metadata.json")
OUTPUT_PATH = QA_DIR / "step1_structured.json"

MAX_PER_SECTION = 15

# ─── 참조 추출 패턴 ──────────────────────────────────
REF_PATTERNS = {
    "gdc": re.compile(
        r'(?:GDC|General Design Criteri(?:on|a))\s*\(?\s*(?:GDC\s*)?(\d+)',
        re.IGNORECASE,
    ),
    "cfr": re.compile(
        r'10\s*CFR\s*(?:Part\s*)?([\d]+(?:\.[\d]+[a-z]*(?:\([a-z0-9]+\))*)*)',
        re.IGNORECASE,
    ),
    "rg": re.compile(
        r'(?:Regulatory\s+Guide|RG)\s+([\d]+\.[\d]+)',
        re.IGNORECASE,
    ),
    "srp": re.compile(
        r'(?:SRP|NUREG[- ]0800)\s+(?:Section\s+)?([\d]+\.[\d]+(?:\.[\d]+)*)',
        re.IGNORECASE,
    ),
    "dsrs": re.compile(
        r'DSRS\s+([\d]+\.[\d]+(?:\.[\d]+)*)',
        re.IGNORECASE,
    ),
}

# ─── core_question 분리 마커 ─────────────────────────
QUESTION_MARKERS = [
    r'Information Request(?:ed)?[\s:\n]',
    r'Requested Information[\s:\n]',
    r'The (?:NRC )?staff request[s]?\s',
    r'Please (?:provide|explain|clarify|justify|describe|address|discuss)\s',
    r'Provide the following',
    r'Provide clarification',
    r'Provide a (?:description|justification|basis|summary)',
    r'The applicant (?:should|is requested|is asked|needs to|must)\s',
    r'(?:Clarify|Explain|Justify|Describe|Address|Discuss|Identify|Confirm|Verify|Update|Revise)\s',
    r'(?:In (?:light|view|consideration) of|Based on|Given) (?:the|this)',
    r'(?:In (?:FSAR|DCD|the DCA))',
    r'(?:FSAR|DCD) (?:Tier|Section|Table|Figure)',
    r'(?:NuScale|The applicant) (?:should|is|has|did not|does not)',
    r'(?:It appears|It is unclear|It is not clear|The staff notes)',
    r'(?:a|b|c|1|2|3)\.\s+(?:Provide|Explain|Clarify|Describe|Identify|Justify|Address)',
]
MARKER_PATTERN = re.compile(
    r'(?:^|\n)\s*(' + '|'.join(QUESTION_MARKERS) + r')',
    re.MULTILINE | re.IGNORECASE,
)


def load_doc_metadata() -> dict:
    """문서 메타데이터 로드 → section_number → doc_id 매핑."""
    with open(META_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    section_to_doc = defaultdict(list)
    for d in docs:
        key = f"{d['doc_type']}:{d['section_number']}"
        section_to_doc[key].append(d['doc_id'])

    # section_number만으로도 찾을 수 있게 (SRP/DSRS 양쪽)
    section_any = defaultdict(list)
    for d in docs:
        if d['doc_type'] in ('SRP', 'DSRS') and d['section_number']:
            section_any[d['section_number']].append(d['doc_id'])

    return section_to_doc, section_any


def load_qa_datasets() -> list[dict]:
    """두 데이터셋을 로드하여 source 표시 후 합침."""
    all_items = []

    for fname, source in [
        ("nuscale_rai_qa_dataset.json", "nuscale"),
        ("levy_rai_qa_dataset.json", "levy"),
    ]:
        path = QA_DIR / fname
        if not path.exists():
            print(f"[WARN] {path} not found, skipping")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data['data']:
            item['_source'] = source
            all_items.append(item)

    return all_items


# ─── Step 0: 필터링 ──────────────────────────────────
def step0_filter(items: list, section_any: dict) -> list:
    """필터링: relevant_docs 없는 것 제외, 섹션 매칭 검증, 섹션당 상한."""
    filtered = []
    stats = {"no_relevant_docs": 0, "no_section_match": 0, "section_capped": 0}
    section_counts = Counter()

    for item in items:
        # 1. relevant_docs 없으면 제외
        if not item.get('relevant_docs'):
            stats["no_relevant_docs"] += 1
            continue

        # 2. section_number가 문서에 매칭 안 되면 제외
        section = item.get('section_number', '')
        if section and section not in section_any:
            stats["no_section_match"] += 1
            continue

        # 3. 섹션당 상한
        if section:
            section_counts[section] += 1
            if section_counts[section] > MAX_PER_SECTION:
                stats["section_capped"] += 1
                continue

        filtered.append(item)

    print(f"[Step 0] 필터링 결과:")
    print(f"  입력: {len(items)}개")
    print(f"  relevant_docs 없음: {stats['no_relevant_docs']}개 제외")
    print(f"  섹션 매칭 실패: {stats['no_section_match']}개 제외")
    print(f"  섹션 상한 초과: {stats['section_capped']}개 제외")
    print(f"  통과: {len(filtered)}개")

    return filtered


# ─── Step 1: 구조 분리 ───────────────────────────────
def step1_extract(items: list, section_any: dict) -> list:
    """구조 분리: background/core 분리, 참조 추출, 난이도 분류."""
    results = []

    for i, item in enumerate(items):
        question = item.get('question', '')
        answer = item.get('answer', '')
        section = item.get('section_number', '')
        source = item.get('_source', '')

        # ── 참조 추출 ──────────────────────────────
        combined_text = question + " " + answer
        refs = {}
        for ref_type, pattern in REF_PATTERNS.items():
            matches = sorted(set(pattern.findall(combined_text)))
            if matches:
                refs[ref_type] = matches

        # ── regulatory_background / core_question 분리 ──
        bg, core = split_question(question)

        # ── 대상 문서 매칭 ──────────────────────────
        matched_docs = []
        if section:
            for doc_id in section_any.get(section, []):
                matched_docs.append(doc_id)

        # ── 난이도 분류 ─────────────────────────────
        n_ref_types = sum(1 for v in refs.values() if v)
        n_total_refs = sum(len(v) for v in refs.values())
        n_relevant_docs = len(item.get('relevant_docs', []))

        if n_total_refs <= 2 and n_relevant_docs <= 1:
            difficulty = "easy"
        elif n_total_refs >= 5 or n_relevant_docs >= 3:
            difficulty = "hard"
        else:
            difficulty = "medium"

        result = {
            "id": f"{source}_{i:04d}",
            "source": source,
            "original_rai_number": item.get('nrc_rai_number', ''),
            "section_number": section,
            "srp_section": item.get('srp_section', ''),
            "difficulty": difficulty,
            "regulatory_background": bg,
            "core_question": core,
            "original_question": question,
            "original_answer": answer,
            "extracted_refs": refs,
            "matched_docs": matched_docs,
            "relevant_docs": item.get('relevant_docs', []),
            "n_total_refs": n_total_refs,
        }
        results.append(result)

    # ── 통계 ─────────────────────────────────────
    diff_counts = Counter(r['difficulty'] for r in results)
    source_counts = Counter(r['source'] for r in results)
    has_core = sum(1 for r in results if r['core_question'] != r['original_question'])

    print(f"\n[Step 1] 구조 분리 결과:")
    print(f"  총: {len(results)}개")
    print(f"  소스: {dict(source_counts)}")
    print(f"  난이도: Easy={diff_counts['easy']}, Medium={diff_counts['medium']}, Hard={diff_counts['hard']}")
    print(f"  bg/core 분리 성공: {has_core}/{len(results)} ({has_core/len(results)*100:.1f}%)")

    return results


def split_question(question: str) -> tuple[str, str]:
    """질문 텍스트에서 regulatory_background와 core_question을 분리한다."""
    # 1차: 명시적 마커 패턴 검색
    match = MARKER_PATTERN.search(question)
    if match:
        bg = question[:match.start()].strip()
        core = question[match.start():].strip()
        if len(core) >= 50:
            return bg, core

    # 2차: "Regulatory Background" / "Regulatory Basis" 헤더
    header_match = re.match(
        r'\s*(Regulatory\s+(?:Background|Basis)\s*:?\s*\n)',
        question, re.IGNORECASE,
    )
    if header_match:
        after_header = question[header_match.end():]
        # 헤더 이후에서 마커 재검색
        match2 = MARKER_PATTERN.search(after_header)
        if match2:
            bg = question[:header_match.end() + match2.start()].strip()
            core = after_header[match2.start():].strip()
            if len(core) >= 50:
                return bg, core

    # 3차: 규제 조항으로 시작하는 경우 (10 CFR, GDC, SRP 등)
    # 규제 배경 서술 후 실제 질문이 나오는 패턴
    reg_start = re.match(
        r'\s*(?:10\s*CFR|GDC|General Design|SRP Section|Standard Review|'
        r'To meet the requirements|Regulatory [Bb]asis)',
        question,
    )
    if reg_start:
        # 문장 단위로 분석 — 질문형 문장을 찾음
        # "?" 가 있는 첫 문장, 또는 명령형 동사가 있는 문장
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', question)
        for j, sent in enumerate(sentences):
            if j == 0:
                continue
            # 질문 마커가 있는 문장
            if MARKER_PATTERN.search(sent):
                bg = ' '.join(sentences[:j]).strip()
                core = ' '.join(sentences[j:]).strip()
                if len(core) >= 50:
                    return bg, core
                break

    # 4차: 마지막 시도 — 텍스트 후반 40%를 core로 간주
    # (규제 배경이 앞부분, 질문이 뒷부분인 구조)
    if len(question) > 300:
        # 60% 지점 이후에서 줄바꿈 기준으로 분리
        split_point = int(len(question) * 0.5)
        newline_pos = question.find('\n', split_point)
        if newline_pos > 0:
            bg = question[:newline_pos].strip()
            core = question[newline_pos:].strip()
            if len(core) >= 80:
                return bg, core

    # 분리 실패 → 전체가 core
    return "", question


def main():
    print("=" * 60)
    print("RAG QA Pipeline — Step 0+1: 필터링 + 구조 분리")
    print("=" * 60)

    # 메타데이터 로드
    section_to_doc, section_any = load_doc_metadata()
    print(f"문서 메타데이터: {sum(len(v) for v in section_any.values())}개 섹션 매핑")

    # QA 데이터셋 로드
    items = load_qa_datasets()
    print(f"QA 데이터 로드: {len(items)}개")

    # Step 0: 필터링
    print()
    filtered = step0_filter(items, section_any)

    # Step 1: 구조 분리
    results = step1_extract(filtered, section_any)

    # 저장
    output = {
        "metadata": {
            "description": "RAG QA Pipeline Step 1 — Structured RAI data",
            "total": len(results),
            "difficulty_distribution": dict(Counter(r['difficulty'] for r in results)),
            "source_distribution": dict(Counter(r['source'] for r in results)),
        },
        "data": results,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUTPUT_PATH}")

    # 난이도별 샘플
    for diff in ['easy', 'medium', 'hard']:
        samples = [r for r in results if r['difficulty'] == diff]
        if samples:
            s = samples[0]
            print(f"\n=== {diff.upper()} 샘플 ===")
            print(f"  RAI: {s['original_rai_number']} (section {s['section_number']})")
            print(f"  refs: {s['extracted_refs']}")
            print(f"  core_q: {s['core_question'][:120]}...")


if __name__ == "__main__":
    main()
