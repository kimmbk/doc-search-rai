"""
Attribute Layer Environment

Document Layer에서 선택된 문서들의 PageIndex 트리를 로드하고
browse/read/search 도구를 제공한다.

데이터 소스: /home/bbo/Document_pageindex/pageindex/{doc_id}_structure.json
포맷: {"doc_name", "structure": [{"title", "start_index", "end_index", "node_id", "summary", "nodes": [...]}]}
"""

import json
import os
from pathlib import Path
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class AttributeLayerEnv:
    """
    선택된 문서들의 PageIndex 트리를 로드하여 탐색 도구를 제공한다.
    """

    def __init__(self, trees_dir: str):
        self.trees_dir = Path(trees_dir)
        self.documents: dict[str, dict] = {}    # doc_id → tree_data
        self.node_cache: dict[str, dict] = {}   # "doc_id::node_id" → node
        self.parent_map: dict[str, str] = {}    # "doc_id::node_id" → parent cache_key
        self._bm25_index = None

    # ─── 문서 로드 ────────────────────────────────────
    def load_documents(self, doc_ids: list[str]) -> None:
        """선택된 문서들의 트리를 로드한다."""
        self.documents.clear()
        self.node_cache.clear()
        self.parent_map.clear()
        self._bm25_index = None

        for doc_id in doc_ids:
            tree_path = self.trees_dir / f"{doc_id}_structure.json"
            if not tree_path.exists():
                continue
            with open(tree_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)

            self.documents[doc_id] = tree_data
            self._cache_nodes(doc_id, tree_data.get("structure", []))

    def _cache_nodes(self, doc_id: str, nodes: list, depth: int = 0,
                     parent_key: str = "") -> None:
        for node in nodes:
            nid = node.get("node_id", "")
            if nid:
                cache_key = f"{doc_id}::{nid}"
                self.node_cache[cache_key] = {
                    **node,
                    "doc_id": doc_id,
                    "depth": depth,
                }
                if parent_key:
                    self.parent_map[cache_key] = parent_key
            children = node.get("nodes", [])
            if children:
                self._cache_nodes(
                    doc_id, children, depth + 1,
                    parent_key=f"{doc_id}::{nid}" if nid else parent_key,
                )

    # ─── browse ───────────────────────────────────────
    def browse(self, doc_id: str = None, node_id: str = None,
               depth: int = 1) -> list[dict]:
        """노드의 자식을 탐색한다."""
        if doc_id is None:
            results = []
            for did, doc in self.documents.items():
                for node in doc.get("structure", []):
                    results.extend(self._browse_recursive(did, node, depth, 0))
            return results

        if doc_id not in self.documents:
            return []

        if node_id is None:
            results = []
            for n in self.documents[doc_id].get("structure", []):
                results.extend(self._browse_recursive(doc_id, n, depth, 0))
            return results

        cache_key = f"{doc_id}::{node_id}"
        node = self.node_cache.get(cache_key)
        if not node:
            return []

        results = []
        for child in node.get("nodes", []):
            results.extend(self._browse_recursive(doc_id, child, depth, 0))
        return results

    def _browse_recursive(self, doc_id, node, max_depth, current_depth):
        results = [{
            "doc_id": doc_id,
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "start_index": node.get("start_index", 0),
            "end_index": node.get("end_index", 0),
            "n_children": len(node.get("nodes", [])),
            "depth": current_depth,
        }]
        if current_depth < max_depth - 1:
            for child in node.get("nodes", []):
                results.extend(self._browse_recursive(
                    doc_id, child, max_depth, current_depth + 1))
        return results

    # ─── read ─────────────────────────────────────────
    def read(self, doc_id: str, node_id: str) -> Optional[dict]:
        """노드의 전체 내용을 읽는다."""
        cache_key = f"{doc_id}::{node_id}"
        node = self.node_cache.get(cache_key)
        if not node:
            return None

        content = node.get("summary", "")

        return {
            "doc_id": doc_id,
            "node_id": node_id,
            "title": node.get("title", ""),
            "content": content,
            "page_range": f"{node.get('start_index', '?')}-{node.get('end_index', '?')}",
            "n_children": len(node.get("nodes", [])),
        }

    # ─── search (BM25) ───────────────────────────────
    def search(self, keyword: str, max_results: int = 5) -> list[dict]:
        """BM25 기반 키워드 검색."""
        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._bm25_keys:
            return []

        query_tokens = keyword.lower().split()

        if HAS_BM25 and self._bm25_index != "tfidf":
            scores = self._bm25_index.get_scores(query_tokens)
        else:
            scores = []
            for text in self._corpus_texts:
                score = sum(text.count(token) for token in query_tokens)
                scores.append(float(score))

        scored = sorted(zip(scores, self._bm25_keys), reverse=True)

        results = []
        for score, cache_key in scored:
            if score <= 0 or len(results) >= max_results:
                break

            node = self.node_cache[cache_key]
            title = node.get("title", "")
            summary = node.get("summary", "")

            snippet = ""
            keyword_lower = keyword.lower()
            if keyword_lower in title.lower():
                snippet = title
            elif keyword_lower in summary.lower():
                idx = summary.lower().find(keyword_lower)
                start = max(0, idx - 80)
                snippet = summary[start:idx + len(keyword) + 80]
            else:
                snippet = title

            results.append({
                "doc_id": node.get("doc_id", ""),
                "node_id": node.get("node_id", ""),
                "title": title,
                "start_index": node.get("start_index", 0),
                "end_index": node.get("end_index", 0),
                "score": round(float(score), 2),
                "snippet": snippet.strip(),
            })

        return results

    def _build_bm25_index(self) -> None:
        self._bm25_keys = []
        self._corpus_texts = []
        corpus = []
        for cache_key, node in self.node_cache.items():
            title = node.get("title", "")
            summary = node.get("summary", "") or ""
            combined = f"{title} {title} {title} {summary}"
            tokens = combined.lower().split()
            self._bm25_keys.append(cache_key)
            self._corpus_texts.append(combined.lower())
            corpus.append(tokens)

        if corpus and HAS_BM25:
            self._bm25_index = BM25Okapi(corpus)
        else:
            self._bm25_index = "tfidf"

    # ─── 개요 ─────────────────────────────────────────
    def get_overview(self, depth: int = 2) -> str:
        """로드된 문서들의 구조 개요를 반환한다."""
        lines = [f"=== Loaded Documents ({len(self.documents)}) ==="]
        for doc_id, doc in self.documents.items():
            lines.append(f"\n[{doc_id}] ({doc.get('doc_name', '?')})")
            items = self.browse(doc_id, None, depth=depth)
            for item in items:
                indent = "  " * (item["depth"] + 1)
                children = f" [{item['n_children']} sub]" if item["n_children"] > 0 else ""
                lines.append(f"{indent}[{item['node_id']}] {item['title'][:60]}{children}")
        return "\n".join(lines)

    @property
    def node_count(self) -> int:
        return len(self.node_cache)

    @property
    def doc_count(self) -> int:
        return len(self.documents)
