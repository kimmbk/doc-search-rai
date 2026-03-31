"""
RAG 쿼리 실행 스크립트

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/query.py "GDC 38 containment heat removal"
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/query.py --search-only "GDC 38 containment heat removal"
"""

import sys
import argparse

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="질문")
    parser.add_argument("--search-only", action="store_true", help="검색만 (LLM 호출 안 함)")
    args = parser.parse_args()

    query = " ".join(args.query)

    if args.search_only:
        from src.pipeline.two_layer_search import build_pipeline
        pipeline = build_pipeline()
        result = pipeline.search(query)
        print("\n".join(result.trace))
        print()
        print(result.context)
    else:
        from src.pipeline.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        result = rag.ask(query)
        print("\n".join(result.trace))
        print()
        print(result.answer)


if __name__ == "__main__":
    main()
