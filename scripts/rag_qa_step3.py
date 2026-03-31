"""
RAG QA Pipeline — Step 3: Ground Truth 추출

입력: step2_queries.json + 문서 PDF (156개)
출력: step3_with_gt.json

각 쿼리에 대해:
  1. section_number → 해당 SRP/DSRS PDF 직접 읽기 (KG 안 씀 — 순환 방지)
  2. LLM: 문서 텍스트에서 쿼리에 대한 정답 추출
  3. evidence_sections, evidence_pages 기록

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/rag_qa_step3.py [--start N] [--end N] [--dry-run]
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("/home/bbo/RAG_Baseline/.env")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── 경로 ─────────────────────────────────────────────
QA_DIR = Path("/home/bbo/RAG_Baseline/qadataset")
DOCS_DIR = Path("/home/bbo/RAG_Baseline/documents")
META_PATH = Path("/home/bbo/doc_search/data/doc_metadata.json")
INPUT_PATH = QA_DIR / "step2_queries.json"
OUTPUT_PATH = QA_DIR / "step3_with_gt.json"

# ─── LLM 설정 ─────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"

# ─── 문서 메타데이터 → section→파일 매핑 ──────────────
def build_section_to_file() -> dict:
    """section_number → {doc_type, filename, pdf_path} 매핑."""
    with open(META_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    mapping = defaultdict(list)
    for d in docs:
        if d['doc_type'] in ('SRP', 'DSRS') and d['section_number']:
            mapping[d['section_number']].append({
                "doc_type": d['doc_type'],
                "doc_id": d['doc_id'],
                "filename": d['filename'],
                "pdf_path": d['pdf_path'],
                "title": d['title'],
            })
    return mapping


# ─── PDF 텍스트 추출 (캐시) ───────────────────────────
_pdf_cache = {}

def read_pdf_text(pdf_path: str, max_pages: int = 30) -> str:
    """PDF 전체 텍스트를 추출한다. 캐시 사용."""
    if pdf_path in _pdf_cache:
        return _pdf_cache[pdf_path]

    try:
        doc = fitz.open(pdf_path)
        pages = min(doc.page_count, max_pages)
        text_parts = []
        for i in range(pages):
            text_parts.append(doc[i].get_text())
        doc.close()
        full_text = "\n".join(text_parts)
        _pdf_cache[pdf_path] = full_text
        return full_text
    except Exception as e:
        return f"[ERROR reading PDF: {e}]"


def find_relevant_section(text: str, query: str) -> tuple[str, str]:
    """문서 텍스트에서 쿼리와 가장 관련 있는 섹션을 찾는다."""
    # NRC 문서의 표준 섹션 구분
    section_patterns = [
        r'(I+V?\.?\s+AREAS OF REVIEW.*?)(?=\nI+V?\.?\s+|\Z)',
        r'(I+V?\.?\s+ACCEPTANCE CRITERIA.*?)(?=\nI+V?\.?\s+|\Z)',
        r'(I+V?\.?\s+REVIEW PROCEDURES.*?)(?=\nI+V?\.?\s+|\Z)',
        r'(I+V?\.?\s+EVALUATION FINDINGS.*?)(?=\nI+V?\.?\s+|\Z)',
        r'(I+V?\.?\s+IMPLEMENTATION.*?)(?=\nI+V?\.?\s+|\Z)',
        r'(I+V?\.?\s+REFERENCES.*?)(?=\nI+V?\.?\s+|\Z)',
        r'(A\.\s+INTRODUCTION.*?)(?=\n[A-Z]\.\s+|\Z)',
        r'(B\.\s+DISCUSSION.*?)(?=\n[A-Z]\.\s+|\Z)',
        r'(C\.\s+STAFF REGULATORY GUIDANCE.*?)(?=\n[A-Z]\.\s+|\Z)',
    ]

    sections = []
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        for m in matches:
            if len(m) > 100:
                # 섹션 제목 추출
                title_match = re.match(r'([IVX]+\.?\s+[A-Z][A-Z\s]+|[A-Z]\.\s+[A-Z][A-Z\s]+)', m)
                title = title_match.group(1).strip() if title_match else "Unknown"
                sections.append((title, m))

    if not sections:
        # 섹션 구분 못 찾으면 전체 텍스트 사용
        return "Full Document", text[:8000]

    # 쿼리 키워드로 가장 관련 높은 섹션 선택
    query_words = set(query.lower().split())
    best_section = None
    best_score = -1

    for title, content in sections:
        content_lower = content.lower()
        score = sum(1 for w in query_words if w in content_lower and len(w) > 3)
        if score > best_score:
            best_score = score
            best_section = (title, content)

    if best_section:
        return best_section

    # acceptance criteria 우선
    for title, content in sections:
        if 'ACCEPTANCE' in title.upper():
            return title, content

    return sections[0]


# ─── Ground Truth 추출 프롬프트 ───────────────────────

GT_SYSTEM = """You are a nuclear regulatory expert extracting ground truth answers from NRC review standard documents (SRP, DSRS, Regulatory Guides).

Your task: Given a query and document text, extract the EXACT answer from the document.

Rules:
1. Answer MUST be directly supported by the provided document text — no inference or external knowledge.
2. Quote relevant regulatory text verbatim when possible.
3. If the document does not contain enough information to answer, say so explicitly.
4. Keep the answer concise but complete (100-500 words).
5. Include specific regulatory references (GDC numbers, CFR sections) mentioned in the text.
"""

GT_USER = """Query: {query}

Document: {doc_type} {section_number} — {title}

Document Text (relevant section):
{section_text}

Extract the ground truth answer from this document text. Respond in JSON:
{{
  "ground_truth": "the extracted answer text",
  "evidence_section": "section title where answer was found",
  "confidence": "high|medium|low",
  "answerable": true,
  "reason": "why answerable or not"
}}
"""


def extract_ground_truth(query: str, doc_info: dict, section_title: str,
                         section_text: str) -> dict:
    """LLM으로 문서에서 ground truth를 추출한다."""
    # 텍스트 길이 제한 (토큰 절약)
    truncated = section_text[:6000]

    prompt = GT_USER.format(
        query=query,
        doc_type=doc_info.get("doc_type", ""),
        section_number=doc_info.get("section_number", ""),
        title=doc_info.get("title", ""),
        section_text=truncated,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": GT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        return json.loads(cleaned)
    except Exception as e:
        return {
            "ground_truth": "",
            "evidence_section": "",
            "confidence": "low",
            "answerable": False,
            "reason": f"LLM error: {str(e)}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG QA Pipeline — Step 3: Ground Truth 추출")
    print("=" * 60)

    # 매핑 로드
    section_map = build_section_to_file()
    print(f"섹션→파일 매핑: {len(section_map)}개")

    # Step 2 결과 로드
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rai_items = data['data']
    end = args.end or len(rai_items)
    rai_items = rai_items[args.start:end]
    print(f"처리 대상 RAI: {len(rai_items)}개 (index {args.start}~{end-1})")

    if args.dry_run:
        print("\n[DRY RUN] 첫 2개 RAI만 처리")
        rai_items = rai_items[:2]

    # 기존 결과 로드 (이어하기)
    existing = []
    if OUTPUT_PATH.exists() and not args.dry_run:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing = existing_data.get("data", [])
        existing_ids = {e["id"] for e in existing}
        print(f"기존 결과: {len(existing)}개")
    else:
        existing_ids = set()

    results = list(existing)
    total_gt = sum(
        sum(1 for q in r.get("queries_with_gt", []) if q.get("ground_truth"))
        for r in results
    )
    not_answerable = 0

    for i, rai in enumerate(rai_items):
        rai_id = rai["id"]
        if rai_id in existing_ids:
            continue

        section = rai.get("section_number", "")
        queries = rai.get("generated_queries", [])

        if not queries:
            results.append({**rai, "queries_with_gt": []})
            continue

        # 대상 문서 찾기
        doc_files = section_map.get(section, [])
        if not doc_files:
            # relevant_docs에서 파일명으로 직접 시도
            for rd in rai.get("relevant_docs", []):
                path_str = rd.get("path", "")
                fname = os.path.basename(path_str)
                full_path = DOCS_DIR / fname
                if full_path.exists():
                    doc_files.append({
                        "doc_type": rd.get("type", ""),
                        "filename": fname,
                        "pdf_path": str(full_path),
                        "title": rd.get("reference", ""),
                        "section_number": section,
                    })

        if not doc_files:
            results.append({**rai, "queries_with_gt": []})
            continue

        # 문서 텍스트 읽기 (첫 번째 매칭 문서)
        doc_info = doc_files[0]
        pdf_path = doc_info.get("pdf_path", "")
        if not pdf_path or not os.path.isfile(pdf_path):
            # DOCS_DIR에서 찾기
            pdf_path = str(DOCS_DIR / doc_info["filename"])

        doc_text = read_pdf_text(pdf_path)

        # 각 쿼리에 대해 GT 추출
        queries_with_gt = []
        for q in queries:
            query_text = q.get("query", "")

            # 관련 섹션 찾기
            sec_title, sec_text = find_relevant_section(doc_text, query_text)

            # LLM으로 GT 추출
            gt_result = extract_ground_truth(
                query=query_text,
                doc_info={**doc_info, "section_number": section},
                section_title=sec_title,
                section_text=sec_text,
            )

            entry = {
                **q,
                "ground_truth": gt_result.get("ground_truth", ""),
                "evidence_doc": doc_info["filename"],
                "evidence_section": gt_result.get("evidence_section", sec_title),
                "confidence": gt_result.get("confidence", "low"),
                "answerable": gt_result.get("answerable", False),
            }
            queries_with_gt.append(entry)

            if gt_result.get("answerable") and gt_result.get("ground_truth"):
                total_gt += 1
            else:
                not_answerable += 1

            time.sleep(0.3)

        results.append({
            "id": rai_id,
            "source": rai.get("source", ""),
            "original_rai_number": rai.get("original_rai_number", ""),
            "section_number": section,
            "srp_section": rai.get("srp_section", ""),
            "difficulty": rai.get("difficulty", ""),
            "extracted_refs": rai.get("extracted_refs", {}),
            "matched_docs": rai.get("matched_docs", []),
            "relevant_docs": rai.get("relevant_docs", []),
            "queries_with_gt": queries_with_gt,
        })

        if (i + 1) % 10 == 0 or i == len(rai_items) - 1:
            print(f"  [{i+1}/{len(rai_items)}] gt={total_gt}, not_answerable={not_answerable}")
            if not args.dry_run:
                _save(results, total_gt, not_answerable)

    _save(results, total_gt, not_answerable)

    # 최종 통계
    all_queries = []
    for r in results:
        for q in r.get("queries_with_gt", []):
            all_queries.append(q)

    answerable = [q for q in all_queries if q.get("answerable") and q.get("ground_truth")]
    conf_counts = Counter(q.get("confidence", "?") for q in answerable)

    print(f"\n=== 최종 결과 ===")
    print(f"  총 쿼리: {len(all_queries)}개")
    print(f"  GT 추출 성공 (answerable): {len(answerable)}개")
    print(f"  GT 추출 실패: {len(all_queries) - len(answerable)}개")
    print(f"  confidence: {dict(conf_counts)}")

    if answerable:
        s = answerable[0]
        print(f"\n  [샘플] Q: \"{s['query'][:80]}...\"")
        print(f"         A: \"{s['ground_truth'][:120]}...\"")
        print(f"         doc: {s['evidence_doc']}, section: {s['evidence_section']}")


def _save(results, total_gt, not_answerable):
    output = {
        "metadata": {
            "description": "RAG QA Pipeline Step 3 — Queries with ground truth",
            "total_rai": len(results),
            "total_gt": total_gt,
            "not_answerable": not_answerable,
        },
        "data": results,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
