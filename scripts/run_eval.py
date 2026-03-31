"""
doc_search RAG 평가 스크립트

step3_with_gt.json의 쿼리를 2-Layer 파이프라인으로 검색 + GPT-4.1 답변 생성 후
RAG_Baseline/04_evaluate.py와 동일한 포맷으로 결과를 저장한다.

출력 포맷: RAG_Baseline/results/query_results.json과 동일
  → 04_evaluate.py로 바로 평가 가능

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/run_eval.py [--n 10] [--all] [--search-only]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")

from openai import OpenAI
from dotenv import load_dotenv
from src.pipeline.two_layer_search import build_pipeline

load_dotenv("/home/bbo/RAG_Baseline/.env")

# ─── 경로 ─────────────────────────────────────────────
QA_DIR = Path("/home/bbo/RAG_Baseline/qadataset")
GT_PATH = QA_DIR / "rag_qa_dataset_final.json"
RESULTS_DIR = Path("/home/bbo/RAG_Baseline/results")
EXPERIMENT_ID = "doc_search_2layer"
EXPERIMENT_DIR = RESULTS_DIR / EXPERIMENT_ID

# ─── LLM 설정 ─────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"

SYSTEM_PROMPT = """You are an NRC nuclear regulatory expert. Answer the question based ONLY on the provided context documents. If the context does not contain enough information, say so."""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def generate_answer(query: str, context: str) -> tuple[str, float]:
    """LLM 답변 생성. (answer, generation_time) 반환."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[ERROR] {str(e)}"
    gen_time = time.time() - start
    return answer, gen_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="테스트할 쿼리 수")
    parser.add_argument("--all", action="store_true", help="전체 쿼리 실행")
    parser.add_argument("--search-only", action="store_true", help="검색만 (LLM 호출 안 함)")
    args = parser.parse_args()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"doc_search 2-Layer RAG Evaluation")
    print(f"  Generation LLM: {MODEL}")
    print("=" * 60)

    # 파이프라인 로드
    print("Loading 2-Layer pipeline...")
    pipeline = build_pipeline()
    print(f"  KG: {len(pipeline.kg.docs)} docs")

    # GT 데이터 로드 → flat query list
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    all_queries = []
    for item in gt_data["data"]:
        if not item.get("ground_truth") or not item.get("answerable"):
            continue
        # evidence_docs에서 doc_id 추출 (확장자 제거)
        evidence_doc_ids = []
        for ed in item.get("evidence_docs", []):
            doc_id = ed.replace(".pdf", "")
            evidence_doc_ids.append(doc_id)
        all_queries.append({
            "query_id": item.get("id", ""),
            "query": item["query"],
            "gt_answer": item["ground_truth"],
            "relevant_docs": item.get("relevant_docs", []),
            "matched_docs": evidence_doc_ids,
            "difficulty": item.get("difficulty", "?"),
        })

    total = len(all_queries)
    print(f"Total queries with GT: {total}")

    if not args.all:
        # 난이도 + 섹션별 균형 샘플링
        by_section = defaultdict(list)
        for q in all_queries:
            sec = q["query_id"].split("_")[1] if "_" in q["query_id"] else "other"
            by_section[sec].append(q)
        sample = []
        sections = sorted(by_section.keys())
        while len(sample) < args.n and sections:
            for sec in list(sections):
                if by_section[sec]:
                    sample.append(by_section[sec].pop(0))
                    if len(sample) >= args.n:
                        break
                else:
                    sections.remove(sec)
        all_queries = sample
    print(f"Running: {len(all_queries)} queries")

    # 출력 경로
    if args.all:
        output_path = EXPERIMENT_DIR / "query_results_full.json"
    else:
        output_path = EXPERIMENT_DIR / "query_results.json"

    # Resume
    results = []
    done_ids = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_ids = {r["query_id"] for r in results}
        print(f"Resuming: {len(done_ids)} done, {len(all_queries) - len(done_ids)} remaining")

    run_count = 0
    for i, q in enumerate(all_queries, 1):
        if q["query_id"] in done_ids:
            continue

        print(f"\n[{i}/{len(all_queries)}] [{q['difficulty'].upper()}] {q['query'][:80]}...")

        # 2-Layer 검색
        start = time.time()
        search_result = pipeline.search(q["query"])
        retrieval_time = time.time() - start

        for line in search_result.trace:
            print(f"  {line}")

        # retrieved_chunks를 baseline 포맷으로 변환
        retrieved_chunks = []
        for rank, section in enumerate(search_result.retrieved_sections, 1):
            retrieved_chunks.append({
                "chunk_id": f"{section['doc_id']}_{section.get('node_id', '')}",
                "doc_id": section["doc_id"],
                "section_path": section.get("title", ""),
                "score": 1.0 / rank,  # rank 기반 score
                "rank": rank,
                "text": section.get("content", section.get("summary", ""))[:300],
            })

        # LLM 답변 생성
        if args.search_only:
            answer = "[SEARCH ONLY]"
            gen_time = 0.0
        else:
            answer, gen_time = generate_answer(q["query"], search_result.context)

        # relevant_docs에 matched_docs 기반 path 추가
        relevant_docs = list(q["relevant_docs"])
        for doc_id in q["matched_docs"]:
            if not any(doc_id in d.get("path", "") for d in relevant_docs):
                relevant_docs.append({"path": doc_id, "reference": doc_id})

        result = {
            "query_id": q["query_id"],
            "query": q["query"],
            "answer": answer,
            "gt_answer": q["gt_answer"],
            "answer_pdf": "",
            "relevant_docs": relevant_docs,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_time": retrieval_time,
            "generation_time": gen_time,
            "top_k": pipeline.max_sections,
            "backend": "doc_search_2layer",
            "n_docs_selected": len(search_result.selected_docs),
        }
        results.append(result)
        run_count += 1

        print(f"  → {len(search_result.selected_docs)} docs, "
              f"{len(retrieved_chunks)} chunks, "
              f"retrieval={retrieval_time:.2f}s, gen={gen_time:.2f}s")
        print(f"  → Answer: {answer[:150]}...")

        # 중간 저장
        if run_count % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.3)

    # 최종 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} results saved to {output_path}")
    print(f"")
    print(f"To evaluate with baseline metrics:")
    print(f"  cd /home/bbo/RAG_Baseline")
    print(f"  python 04_evaluate.py --input ../{output_path.relative_to(Path('/home/bbo/RAG_Baseline'))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
