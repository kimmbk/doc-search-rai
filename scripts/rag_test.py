"""
RAG 파이프라인 테스트: 2-Layer 검색 + LLM 답변 생성

입력: step2_queries.json (965개 쿼리)
출력: rag_test_results.json

Flow:
  1. Query → TwoLayerSearchPipeline (검색 컨텍스트)
  2. Context + Query → GPT-4.1 (답변 생성)

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/rag_test.py [--n 10] [--dry-run]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, ".")

from openai import OpenAI
from dotenv import load_dotenv
from src.pipeline.two_layer_search import build_pipeline

load_dotenv("/home/bbo/RAG_Baseline/.env")

# ─── 경로 ─────────────────────────────────────────────
QA_DIR = Path("/home/bbo/RAG_Baseline/qadataset")
QUERIES_PATH = QA_DIR / "step2_queries.json"
OUTPUT_PATH = QA_DIR / "rag_test_results.json"

# ─── LLM 설정 ─────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"

SYSTEM_PROMPT = """You are an expert on U.S. Nuclear Regulatory Commission (NRC) regulations and review standards.

Answer the user's question using ONLY the retrieved context provided below.
- If the context contains the answer, provide a clear, specific response with regulatory references.
- If the context is insufficient, say so honestly and explain what information is missing.
- Be precise and cite specific sections, criteria, or regulatory guides when possible.
- Answer in English."""

USER_PROMPT = """Retrieved Context:
{context}

Question: {query}

Answer based on the context above:"""


def generate_answer(query: str, context: str) -> dict:
    """검색 컨텍스트 + 쿼리 → LLM 답변 생성."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(context=context, query=query)},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        usage = response.usage
        return {
            "answer": answer,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
            },
        }
    except Exception as e:
        return {"answer": f"[ERROR] {str(e)}", "tokens": {"prompt": 0, "completion": 0}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="테스트할 쿼리 수")
    parser.add_argument("--dry-run", action="store_true", help="검색만, LLM 호출 안 함")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Pipeline Test: 2-Layer Search + GPT-4.1")
    print("=" * 60)

    # 파이프라인 로드
    print("Loading pipeline...")
    pipeline = build_pipeline()
    print(f"  KG: {len(pipeline.kg.docs)} docs")

    # 쿼리 로드
    with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    # 모든 쿼리를 flat list로
    all_queries = []
    for item in query_data["data"]:
        for q in item.get("generated_queries", []):
            all_queries.append({
                "rai_id": item["id"],
                "section": item["section_number"],
                "difficulty": q.get("sub_difficulty", "?"),
                "query": q["query"],
                "target_concepts": q.get("target_concepts", []),
            })

    print(f"Total queries: {len(all_queries)}, testing: {args.n}")

    # 난이도별 균형 샘플링
    by_diff = {"easy": [], "medium": [], "hard": []}
    for q in all_queries:
        d = q["difficulty"]
        if d in by_diff:
            by_diff[d].append(q)

    sample = []
    per_diff = max(1, args.n // 3)
    for d in ["easy", "medium", "hard"]:
        sample.extend(by_diff[d][:per_diff])
    sample = sample[:args.n]

    print(f"Sample: {len([s for s in sample if s['difficulty']=='easy'])} easy, "
          f"{len([s for s in sample if s['difficulty']=='medium'])} medium, "
          f"{len([s for s in sample if s['difficulty']=='hard'])} hard")

    # 실행
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, q in enumerate(sample):
        print(f"\n[{i+1}/{len(sample)}] [{q['difficulty'].upper()}] {q['query'][:80]}...")

        # Layer 1 + Layer 2 검색
        search_result = pipeline.search(q["query"])
        for line in search_result.trace:
            print(f"  {line}")

        if args.dry_run:
            results.append({
                **q,
                "n_docs": len(search_result.selected_docs),
                "n_sections": len(search_result.retrieved_sections),
                "context_len": len(search_result.context),
                "answer": "[DRY RUN]",
            })
            continue

        # LLM 답변 생성
        llm_result = generate_answer(q["query"], search_result.context)
        total_prompt_tokens += llm_result["tokens"]["prompt"]
        total_completion_tokens += llm_result["tokens"]["completion"]

        result = {
            **q,
            "n_docs": len(search_result.selected_docs),
            "n_sections": len(search_result.retrieved_sections),
            "context_len": len(search_result.context),
            "answer": llm_result["answer"],
            "tokens": llm_result["tokens"],
        }
        results.append(result)

        print(f"  → {len(search_result.selected_docs)} docs, "
              f"{len(search_result.retrieved_sections)} sections, "
              f"{len(search_result.context)} chars")
        print(f"  → Answer: {llm_result['answer'][:150]}...")

        time.sleep(0.3)

    # 저장
    output = {
        "metadata": {
            "model": MODEL,
            "total_tested": len(results),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
        "results": results,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} queries tested")
    print(f"Tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print(f"Results: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
