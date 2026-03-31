"""
RAG QA Pipeline — Step 4: 검증

입력: step3_with_gt.json
출력: rag_qa_dataset_final.json

검증 항목:
  1. ground_truth가 비어있지 않은지
  2. answerable=true인지
  3. confidence가 high/medium인지
  4. LLM Judge: query → ground_truth 답변 가능성 판정
  5. 최종 통계 및 불량 항목 필터링

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/rag_qa_step4.py [--skip-judge] [--dry-run]
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
INPUT_PATH = QA_DIR / "step3_with_gt.json"
OUTPUT_PATH = QA_DIR / "rag_qa_dataset_final.json"

# ─── LLM 설정 ─────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4.1"  # 추출 모델과 다른 temperature 사용


# ─── LLM Judge 프롬프트 ───────────────────────────────

JUDGE_SYSTEM = """You are a strict quality judge for a nuclear regulatory RAG benchmark dataset.

Evaluate whether a query-answer pair meets quality standards for a RAG evaluation benchmark.

Scoring criteria:
1. ANSWERABLE: Can the query be fully answered by the ground truth? (not partially)
2. GROUNDED: Is the ground truth clearly from a regulatory document (not fabricated)?
3. SELF-CONTAINED: Does the query make sense without additional context?
4. NON-TRIVIAL: Does answering the query require actually reading the document (not common knowledge)?
"""

JUDGE_PROMPT = """Evaluate this RAG benchmark entry:

Query: {query}
Difficulty: {difficulty}
Ground Truth Answer: {ground_truth}
Evidence Document: {evidence_doc}

Rate as PASS or FAIL. Respond in JSON:
{{
  "verdict": "PASS|FAIL",
  "scores": {{
    "answerable": true,
    "grounded": true,
    "self_contained": true,
    "non_trivial": true
  }},
  "issues": []
}}

If any score is false, verdict must be FAIL and explain in issues.
"""


def judge_entry(query: str, difficulty: str, ground_truth: str,
                evidence_doc: str) -> dict:
    """LLM Judge로 품질 판정."""
    prompt = JUDGE_PROMPT.format(
        query=query,
        difficulty=difficulty,
        ground_truth=ground_truth[:1000],
        evidence_doc=evidence_doc,
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"verdict": "FAIL", "scores": {}, "issues": [f"Judge error: {e}"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true",
                        help="LLM Judge 단계 건너뛰기 (기본 필터만)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG QA Pipeline — Step 4: 검증")
    print("=" * 60)

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rai_items = data['data']
    print(f"입력 RAI: {len(rai_items)}개")

    # ── 모든 쿼리 수집 ────────────────────────────────
    all_entries = []
    for rai in rai_items:
        for q in rai.get("queries_with_gt", []):
            entry = {
                "id": f"{rai['id']}_{q.get('sub_difficulty', 'unknown')}",
                "source": rai.get("source", ""),
                "original_rai_number": rai.get("original_rai_number", ""),
                "section_number": rai.get("section_number", ""),
                "srp_section": rai.get("srp_section", ""),
                "rai_difficulty": rai.get("difficulty", ""),
                "difficulty": q.get("sub_difficulty", rai.get("difficulty", "")),
                "query": q.get("query", ""),
                "ground_truth": q.get("ground_truth", ""),
                "evidence_docs": [q.get("evidence_doc", "")],
                "evidence_section": q.get("evidence_section", ""),
                "confidence": q.get("confidence", ""),
                "answerable": q.get("answerable", False),
                "target_concepts": q.get("target_concepts", []),
                "extracted_refs": rai.get("extracted_refs", {}),
                "relevant_docs": rai.get("relevant_docs", []),
            }
            all_entries.append(entry)

    print(f"총 쿼리: {len(all_entries)}개")

    # ── Step 4a: 기본 필터 ────────────────────────────
    print("\n[4a] 기본 필터...")

    filtered = []
    filter_stats = Counter()

    for entry in all_entries:
        # 필터 1: ground_truth 비어있음
        if not entry.get("ground_truth") or len(entry["ground_truth"]) < 20:
            filter_stats["empty_gt"] += 1
            continue

        # 필터 2: answerable=false
        if not entry.get("answerable"):
            filter_stats["not_answerable"] += 1
            continue

        # 필터 3: confidence=low
        if entry.get("confidence") == "low":
            filter_stats["low_confidence"] += 1
            continue

        # 필터 4: query 너무 짧음
        if len(entry.get("query", "")) < 20:
            filter_stats["short_query"] += 1
            continue

        filtered.append(entry)

    print(f"  통과: {len(filtered)}개")
    for reason, count in filter_stats.most_common():
        print(f"  제외 - {reason}: {count}개")

    # ── Step 4b: LLM Judge ────────────────────────────
    if not args.skip_judge:
        print(f"\n[4b] LLM Judge...")

        if args.dry_run:
            judge_targets = filtered[:5]
            print(f"  [DRY RUN] 5개만 판정")
        else:
            judge_targets = filtered

        passed = []
        failed = 0

        for i, entry in enumerate(judge_targets):
            result = judge_entry(
                query=entry["query"],
                difficulty=entry["difficulty"],
                ground_truth=entry["ground_truth"],
                evidence_doc=entry["evidence_docs"][0] if entry["evidence_docs"] else "",
            )

            entry["judge_verdict"] = result.get("verdict", "FAIL")
            entry["judge_scores"] = result.get("scores", {})
            entry["judge_issues"] = result.get("issues", [])

            if result.get("verdict") == "PASS":
                passed.append(entry)
            else:
                failed += 1

            if (i + 1) % 20 == 0 or i == len(judge_targets) - 1:
                print(f"  [{i+1}/{len(judge_targets)}] PASS={len(passed)}, FAIL={failed}")

            time.sleep(0.2)

        final = passed
        print(f"\n  Judge 통과: {len(passed)}개, 탈락: {failed}개")
    else:
        print("\n[4b] LLM Judge 건너뜀")
        final = filtered
        for entry in final:
            entry["judge_verdict"] = "SKIP"

    # ── ID 재부여 (중복 방지) ─────────────────────────
    id_counter = Counter()
    for entry in final:
        base_id = f"{entry['source']}_{entry['section_number']}_{entry['difficulty']}"
        id_counter[base_id] += 1
        entry["id"] = f"{base_id}_{id_counter[base_id]:03d}"

    # ── 최종 통계 ─────────────────────────────────────
    diff_counts = Counter(e["difficulty"] for e in final)
    source_counts = Counter(e["source"] for e in final)
    conf_counts = Counter(e["confidence"] for e in final)
    section_counts = Counter(e["section_number"] for e in final)

    print(f"\n{'='*60}")
    print(f"=== 최종 RAG QA 데이터셋 ===")
    print(f"{'='*60}")
    print(f"  총: {len(final)}개")
    print(f"  소스: {dict(source_counts)}")
    print(f"  난이도: Easy={diff_counts.get('easy',0)}, "
          f"Medium={diff_counts.get('medium',0)}, "
          f"Hard={diff_counts.get('hard',0)}")
    print(f"  confidence: {dict(conf_counts)}")
    print(f"  섹션 수: {len(section_counts)}개")
    print(f"  섹션 top5: {section_counts.most_common(5)}")

    # ── 저장 ──────────────────────────────────────────
    output = {
        "metadata": {
            "description": "RAG QA Benchmark Dataset for NRC Nuclear Regulatory Documents",
            "total": len(final),
            "difficulty_distribution": dict(diff_counts),
            "source_distribution": dict(source_counts),
            "corpus_size": 156,
            "pipeline": "RAI → Step0(filter) → Step1(extract) → Step2(generate) → Step3(ground_truth) → Step4(validate)",
        },
        "data": final,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUTPUT_PATH}")

    # 샘플 출력
    for diff in ['easy', 'medium', 'hard']:
        samples = [e for e in final if e['difficulty'] == diff]
        if samples:
            s = samples[0]
            print(f"\n  [{diff.upper()}]")
            print(f"    Q: \"{s['query'][:100]}\"")
            print(f"    A: \"{s['ground_truth'][:100]}\"")
            print(f"    doc: {s['evidence_docs']}")


if __name__ == "__main__":
    main()
