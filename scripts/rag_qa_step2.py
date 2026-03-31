"""
RAG QA Pipeline — Step 2: RAG 쿼리 생성

입력: step1_structured.json (509개)
출력: step2_queries.json

각 RAI를 난이도별 RAG 쿼리로 변환:
  Easy:   단일 문서 팩트 조회 (문서 힌트 없는 자연어)
  Medium: 2-3 문서 교차 참조 (규제 개념만 언급)
  Hard:   시나리오 기반 종합 질문

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/rag_qa_step2.py [--start N] [--end N] [--dry-run]
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("/home/bbo/RAG_Baseline/.env")

# ─── 경로 ─────────────────────────────────────────────
QA_DIR = Path("/home/bbo/RAG_Baseline/qadataset")
INPUT_PATH = QA_DIR / "step1_structured.json"
OUTPUT_PATH = QA_DIR / "step2_queries.json"

# ─── LLM 설정 ─────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"

# ─── 프롬프트 템플릿 ──────────────────────────────────

SYSTEM_PROMPT = """You are a nuclear regulatory expert creating RAG benchmark questions.

Your task: Transform an NRC RAI (Request for Additional Information) into clean, natural-language RAG queries that can be answered using ONLY regulatory review documents (SRP, DSRS, Regulatory Guides, 10 CFR).

IMPORTANT RULES:
1. Do NOT reference specific document IDs (e.g., "DSRS 6.2.2", "SRP 3.7.1") in Medium/Hard queries. Use regulatory concepts instead.
2. Do NOT create questions that require FSAR (applicant's safety analysis) to answer.
3. Questions must be answerable from NRC review standards and regulatory guides only.
4. Write questions in English, as a regulatory professional would ask them.
5. Each question must be self-contained — no "refer to above" or "as mentioned".
"""

QUERY_GEN_PROMPT = """Based on this RAI data, generate RAG benchmark queries.

RAI Number: {rai_number}
SRP/DSRS Section: {section_number} ({srp_section})
Difficulty: {difficulty}
Regulatory References: {refs}
Core Question: {core_question}

Generate queries based on difficulty level:

If difficulty=easy: Generate 1 query.
  - Single document fact lookup
  - Ask about ONE specific regulatory requirement or acceptance criterion
  - Do NOT mention document IDs — use topic/concept instead
  - Example: "What are the NRC acceptance criteria for containment heat removal systems?"

If difficulty=medium: Generate 2 queries.
  - Cross-reference between 2-3 regulatory concepts
  - Ask how requirements relate to each other
  - Example: "How do general design criteria for natural phenomena protection apply to seismic design parameters for nuclear power plants?"

If difficulty=hard: Generate 3 queries (1 easy + 1 medium + 1 hard).
  - Easy: Single fact from one regulatory area
  - Medium: Cross-reference between 2 concepts
  - Hard: Multi-document synthesis, scenario-based
  - Example hard: "For a passive small modular reactor, what combined regulatory provisions from design criteria, review standards, and regulatory guides govern the analysis of station blackout events?"

Respond ONLY in JSON:
{{
  "queries": [
    {{
      "query": "the question text",
      "sub_difficulty": "easy|medium|hard",
      "target_concepts": ["concept1", "concept2"],
      "answerable_without_fsar": true
    }}
  ],
  "skip_reason": null
}}

If the RAI cannot produce questions answerable without FSAR, set "queries": [] and explain in "skip_reason".
"""


def generate_queries(item: dict) -> dict:
    """단일 RAI에서 RAG 쿼리를 생성한다."""
    refs_str = json.dumps(item.get("extracted_refs", {}), ensure_ascii=False)
    core_q = item.get("core_question", "")[:1500]  # 토큰 절약

    prompt = QUERY_GEN_PROMPT.format(
        rai_number=item.get("original_rai_number", ""),
        section_number=item.get("section_number", ""),
        srp_section=item.get("srp_section", ""),
        difficulty=item.get("difficulty", "medium"),
        refs=refs_str,
        core_question=core_q,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        result = json.loads(cleaned)
        return result
    except Exception as e:
        return {"queries": [], "skip_reason": f"LLM error: {str(e)}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG QA Pipeline — Step 2: RAG 쿼리 생성")
    print("=" * 60)

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = data['data']
    end = args.end or len(items)
    items = items[args.start:end]
    print(f"처리 대상: {len(items)}개 (index {args.start}~{end-1})")

    if args.dry_run:
        print("\n[DRY RUN] 첫 3개만 처리")
        items = items[:3]

    # 기존 결과 로드 (이어하기)
    existing = []
    if OUTPUT_PATH.exists() and not args.dry_run:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing = existing_data.get("data", [])
        existing_ids = {e["id"] for e in existing}
        items = [it for it in items if it["id"] not in existing_ids]
        print(f"기존 결과: {len(existing)}개, 신규: {len(items)}개")

    results = list(existing)
    skipped = 0
    total_queries = sum(len(r.get("generated_queries", [])) for r in results)

    for i, item in enumerate(items):
        llm_result = generate_queries(item)

        queries = llm_result.get("queries", [])
        skip_reason = llm_result.get("skip_reason")

        # FSAR 필요한 질문 필터
        valid_queries = [
            q for q in queries
            if q.get("answerable_without_fsar", True)
        ]

        entry = {
            "id": item["id"],
            "source": item["source"],
            "original_rai_number": item["original_rai_number"],
            "section_number": item["section_number"],
            "srp_section": item["srp_section"],
            "difficulty": item["difficulty"],
            "extracted_refs": item["extracted_refs"],
            "matched_docs": item["matched_docs"],
            "relevant_docs": item["relevant_docs"],
            "generated_queries": valid_queries,
            "skip_reason": skip_reason,
        }
        results.append(entry)
        total_queries += len(valid_queries)

        if skip_reason:
            skipped += 1

        if (i + 1) % 10 == 0 or i == len(items) - 1:
            print(f"  [{i+1}/{len(items)}] queries={total_queries}, skipped={skipped}")

            # 중간 저장
            if not args.dry_run:
                _save(results, total_queries, skipped)

        # Rate limit
        time.sleep(0.3)

    _save(results, total_queries, skipped)

    # 통계
    all_queries = []
    for r in results:
        for q in r.get("generated_queries", []):
            all_queries.append(q)

    diff_counts = Counter(q.get("sub_difficulty", "?") for q in all_queries)
    print(f"\n=== 최종 결과 ===")
    print(f"  입력 RAI: {len(results)}개")
    print(f"  생성 쿼리: {len(all_queries)}개")
    print(f"  스킵: {skipped}개")
    print(f"  난이도: {dict(diff_counts)}")

    # 샘플
    for diff in ['easy', 'medium', 'hard']:
        samples = [q for q in all_queries if q.get('sub_difficulty') == diff]
        if samples:
            print(f"\n  [{diff.upper()}] \"{samples[0]['query'][:100]}...\"")


def _save(results, total_queries, skipped):
    output = {
        "metadata": {
            "description": "RAG QA Pipeline Step 2 — Generated queries",
            "total_rai": len(results),
            "total_queries": total_queries,
            "skipped": skipped,
        },
        "data": results,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
