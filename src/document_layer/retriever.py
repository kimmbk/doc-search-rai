"""
KAG-style Document Layer Retrievers

3가지 검색을 병렬 실행하고 점수를 합산하여 관련 문서를 선택한다.

1. kg_cs: KG Constraint Search (regex 참조 매칭 → KG 1-2홉)
2. kg_fr: KG Fuzzy Reasoning + Personalized PageRank
3. rc:    BM25 문서 제목/키워드 검색

+ KAGMerger: min-max 정규화 → 가중 합산 → top-k
"""

import re
from collections import defaultdict

import networkx as nx

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


# GDC 주제 설명 (10 CFR 50 Appendix A)
GDC_DESCRIPTIONS = {
    "1": "quality standards and records",
    "2": "design bases for protection against natural phenomena earthquakes seismic tornadoes hurricanes floods tsunami site parameters",
    "3": "fire protection",
    "4": "environmental and dynamic effects design bases missiles pipe whip",
    "5": "sharing of structures systems and components",
    "10": "reactor design",
    "11": "reactor inherent protection",
    "12": "suppression of reactor power oscillations",
    "13": "instrumentation and control",
    "14": "reactor coolant pressure boundary",
    "15": "reactor coolant system design",
    "16": "containment design leaktight barrier",
    "17": "electric power systems",
    "18": "inspection and testing of electric power systems",
    "19": "control room",
    "20": "protection system functions",
    "21": "protection system reliability and testability",
    "22": "protection system independence",
    "23": "protection system failure modes",
    "24": "separation of protection and control systems",
    "25": "protection system requirements for reactivity control malfunctions",
    "26": "reactivity control system redundancy and capability",
    "27": "combined reactivity control systems capability",
    "28": "reactivity limits",
    "29": "protection against anticipated operational occurrences",
    "30": "quality of reactor coolant pressure boundary",
    "31": "fracture prevention of reactor coolant pressure boundary",
    "32": "inspection of reactor coolant pressure boundary",
    "33": "reactor coolant makeup",
    "34": "residual heat removal",
    "35": "emergency core cooling",
    "36": "inspection of emergency core cooling system",
    "37": "testing of emergency core cooling system",
    "38": "containment heat removal",
    "39": "inspection of containment heat removal system",
    "40": "testing of containment heat removal system",
    "41": "containment atmosphere cleanup",
    "42": "inspection of containment atmosphere cleanup systems",
    "43": "testing of containment atmosphere cleanup systems",
    "44": "cooling water structural and equipment cooling",
    "45": "inspection of cooling water system",
    "46": "testing of cooling water system",
    "50": "containment design basis pressure temperature",
    "51": "fracture prevention of containment pressure boundary",
    "52": "capability for containment leakage rate testing",
    "53": "provisions for containment testing and inspection",
    "54": "piping systems penetrating containment",
    "55": "reactor coolant pressure boundary penetrating containment",
    "56": "primary containment isolation",
    "60": "control of releases of radioactive materials to the environment",
    "61": "fuel storage and handling and radioactivity control",
    "62": "prevention of criticality in fuel storage and handling",
    "63": "monitoring fuel and waste storage",
    "64": "monitoring radioactivity releases",
}


# ─── rc: BM25 Document Retriever ─────────────────────────

class DocBM25Retriever:
    """문서 title + display_name + section_number + chapter topic를 BM25로 검색."""

    # SRP/DSRS 챕터별 주제 설명
    CHAPTER_TOPICS = {
        "2": "site characteristics meteorology climatology geology hydrology seismology geography regional local weather precipitation tornado hurricane wind flood tsunami site parameters natural phenomena",
        "3": "design of structures components equipment seismic wind missiles loads structural analysis concrete steel containment foundation",
        "4": "reactor core thermal hydraulic fuel assembly control rods nuclear design",
        "5": "reactor coolant system pressure boundary steam generators pressurizer piping",
        "6": "engineered safety features containment emergency core cooling ECCS heat removal isolation atmosphere cleanup",
        "7": "instrumentation and controls protection system safety system digital I&C",
        "8": "electric power onsite offsite AC DC diesel generator station blackout",
        "9": "auxiliary systems fuel storage cooling water fire protection communications HVAC",
        "10": "steam and power conversion turbine generator feedwater condensate",
        "11": "radioactive waste management liquid gaseous solid effluent",
        "12": "radiation protection ALARA shielding monitoring health physics dose",
        "13": "conduct of operations emergency planning security training procedures",
        "14": "initial test program ITAAC verification startup testing inspections",
        "15": "transient and accident analyses safety analysis LOCA ATWS anticipated events design basis accident",
        "17": "quality assurance program QA",
        "19": "probabilistic risk assessment PRA severe accidents",
    }

    def __init__(self, docs: dict, graph: nx.DiGraph):
        """
        Args:
            docs: {doc_id: DocumentMeta}
            graph: KG DiGraph (GDC 노드의 title도 인덱싱)
        """
        self._doc_ids = []
        self._corpus_texts = []
        corpus = []

        for doc_id, meta in docs.items():
            text = f"{meta.title} {meta.display_name} {meta.doc_type} {meta.section_number}"
            # 챕터 주제 추가 (섹션번호 첫 자리 기반)
            chapter = meta.section_number.split(".")[0] if meta.section_number else ""
            chapter_topic = self.CHAPTER_TOPICS.get(chapter, "")
            if chapter_topic:
                text += f" {chapter_topic}"
            # GDC refs + 주제 설명 추가
            for gdc in meta.gdc_refs:
                desc = GDC_DESCRIPTIONS.get(gdc, "")
                text += f" GDC {gdc} general design criterion {gdc} {desc}"
            # cross-ref된 섹션 번호도 추가
            for ref in meta.rg_refs:
                text += f" regulatory guide {ref}"
            for ref in meta.srp_refs:
                text += f" SRP {ref}"

            self._doc_ids.append(doc_id)
            self._corpus_texts.append(text.lower())
            corpus.append(text.lower().split())

        if corpus and HAS_BM25:
            self._index = BM25Okapi(corpus)
        else:
            self._index = None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """쿼리로 문서 검색. Returns [(doc_id, score), ...]."""
        tokens = query.lower().split()

        if self._index is not None:
            scores = self._index.get_scores(tokens)
        else:
            # TF-IDF fallback
            scores = []
            for text in self._corpus_texts:
                score = sum(text.count(t) for t in tokens)
                scores.append(float(score))

        results = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(did, s) for did, s in results[:top_k] if s > 0]


# ─── kg_fr: KG Fuzzy Matching + PPR ─────────────────────

class KGFuzzyPPRRetriever:
    """
    쿼리 키워드를 KG 노드에 fuzzy 매칭 → seed 노드에서 PPR 전파.
    """

    def __init__(self, docs: dict, graph: nx.DiGraph):
        self.docs = docs
        self.graph = graph
        # 노드별 검색 텍스트 캐시
        self._node_text: dict[str, str] = {}
        for node, data in graph.nodes(data=True):
            title = data.get("title", "")
            display = data.get("display_name", "")
            text = f"{title} {display}"
            # GDC 노드에 주제 설명 추가
            if node.startswith("gdc:"):
                gdc_num = node.replace("gdc:", "")
                desc = GDC_DESCRIPTIONS.get(gdc_num, "")
                text += f" {desc}"
            # 문서 노드에 GDC 참조 주제 추가
            elif node.startswith("doc:"):
                doc_id = node.replace("doc:", "")
                meta = docs.get(doc_id)
                if meta:
                    for gdc in meta.gdc_refs:
                        desc = GDC_DESCRIPTIONS.get(gdc, "")
                        text += f" {desc}"
            self._node_text[node] = text.lower()

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        쿼리 → seed 노드 fuzzy 매칭 → 두 가지 경로로 문서 점수 산출:
        1. Personalized PageRank (그래프 전파)
        2. Direct entity traversal (seed된 GDC/엔티티를 인용하는 문서에 직접 점수)
        """
        seeds = self._find_seed_nodes(query)
        if not seeds:
            return []

        doc_scores = {}

        # ── Path A: Direct entity → document traversal ────
        # seed된 GDC 엔티티를 직접 인용하는 문서에 높은 점수
        entity_seeds = {k: v for k, v in seeds.items() if not k.startswith("doc:")}
        for entity_node, seed_weight in entity_seeds.items():
            if not self.graph.has_node(entity_node):
                continue
            # 이 엔티티와 연결된 모든 문서 (양방향)
            neighbors = set(self.graph.predecessors(entity_node)) | set(self.graph.successors(entity_node))
            for neighbor in neighbors:
                if neighbor.startswith("doc:"):
                    doc_id = neighbor.replace("doc:", "")
                    if doc_id in self.docs:
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + seed_weight

        # ── Path B: PPR (그래프 전파) ─────────────────────
        personalization = {node: 0.0 for node in self.graph.nodes()}
        for node, weight in seeds.items():
            personalization[node] = weight

        try:
            ppr = nx.pagerank(
                self.graph,
                alpha=0.85,
                personalization=personalization,
                max_iter=100,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            ppr = nx.pagerank(
                self.graph,
                alpha=0.85,
                personalization=personalization,
                max_iter=300,
                tol=1e-4,
            )

        # PPR 점수를 direct traversal 점수에 합산
        for node, score in ppr.items():
            if node.startswith("doc:"):
                doc_id = node.replace("doc:", "")
                if doc_id in self.docs:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _find_seed_nodes(self, query: str) -> dict[str, float]:
        """쿼리 키워드로 KG 노드를 fuzzy 매칭하여 seed 가중치를 반환."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        # 불용어 제거
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "for", "of", "in", "to", "and", "or", "on", "at", "by",
            "with", "from", "that", "this", "how", "what", "which",
            "do", "does", "should", "must", "can", "could", "would",
            "not", "no", "its", "their", "they", "it", "as",
        }
        query_tokens -= stopwords
        if not query_tokens:
            return {}

        seeds = {}
        for node, text in self._node_text.items():
            # 토큰 매칭 점수
            text_tokens = set(text.split())
            overlap = query_tokens & text_tokens
            if not overlap:
                continue

            # 매칭 비율 기반 점수
            score = len(overlap) / len(query_tokens)

            # 연속 구절 매칭 보너스
            for token in overlap:
                if token in query_lower and token in text:
                    score += 0.1

            if score > 0.15:  # 최소 임계값
                seeds[node] = score

        # GDC/엔티티 노드는 항상 포함, 문서 노드는 상위 20개
        entity_seeds = {k: v for k, v in seeds.items() if not k.startswith("doc:")}
        doc_seeds = {k: v for k, v in seeds.items() if k.startswith("doc:")}

        if len(doc_seeds) > 20:
            top = sorted(doc_seeds.items(), key=lambda x: x[1], reverse=True)[:20]
            doc_seeds = dict(top)

        entity_seeds.update(doc_seeds)
        return entity_seeds


# ─── kg_cs: KG Constraint Search (기존 regex 기반) ──────

class KGConstraintRetriever:
    """명시적 참조 패턴(GDC/RG/SRP/DSRS/CFR) 매칭 → KG 1-2홉 탐색."""

    def __init__(self, docs: dict, graph: nx.DiGraph, section_index: dict):
        self.docs = docs
        self.graph = graph
        self.section_index = section_index

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """명시적 참조 매칭. Returns [(doc_id, score), ...]."""
        matched_nodes = set()
        direct_doc_ids = set()

        # GDC
        gdc_matches = re.findall(r'GDC[)\s]+(\d+)', query, re.IGNORECASE)
        gdc_matches += re.findall(
            r'General Design Criteri(?:on|a)\s*\(?GDC\)?\s*(\d+)', query, re.IGNORECASE
        )
        for g in set(gdc_matches):
            node = f"gdc:{g}"
            if self.graph.has_node(node):
                matched_nodes.add(node)

        # RG
        for r in re.findall(r'(?:RG|Regulatory Guide)\s+([\d]+\.[\d]+)', query, re.IGNORECASE):
            for doc_id in self._find_by_section("RG", r):
                direct_doc_ids.add(doc_id)

        # SRP
        for s in re.findall(r'SRP\s+([\d]+\.[\d]+(?:\.[\d]+)*)', query, re.IGNORECASE):
            for doc_id in self._find_by_section("SRP", s):
                direct_doc_ids.add(doc_id)

        # DSRS
        for d in re.findall(r'DSRS\s+([\d]+\.[\d]+(?:\.[\d]+)*)', query, re.IGNORECASE):
            for doc_id in self._find_by_section("DSRS", d):
                direct_doc_ids.add(doc_id)

        # 10 CFR → REG
        if re.findall(r'10\s*CFR', query, re.IGNORECASE):
            for node, data in self.graph.nodes(data=True):
                if data.get("doc_type") == "REG":
                    direct_doc_ids.add(node.replace("doc:", ""))

        # Section 번호 패턴
        for sec in re.findall(r'(?:Section|SECTION)\s+([\d]+\.[\d]+(?:\.[\d]+)*)', query):
            for doc_id in self._find_by_section("SRP", sec):
                direct_doc_ids.add(doc_id)
            for doc_id in self._find_by_section("DSRS", sec):
                direct_doc_ids.add(doc_id)

        # 매칭 노드에서 1-2홉 탐색
        doc_ids = set(direct_doc_ids)
        for node in matched_nodes:
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor].get("node_kind") == "document":
                    doc_ids.add(neighbor.replace("doc:", ""))
                for n2 in self.graph.neighbors(neighbor):
                    if self.graph.nodes[n2].get("node_kind") == "document":
                        doc_ids.add(n2.replace("doc:", ""))

        # 직접 매칭 문서의 이웃도 추가
        for doc_id in list(direct_doc_ids):
            node = f"doc:{doc_id}"
            if self.graph.has_node(node):
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get("node_kind") == "document":
                        doc_ids.add(neighbor.replace("doc:", ""))

        # 점수: 직접 매칭 1.0, 이웃 0.5
        results = []
        for doc_id in doc_ids:
            if doc_id in self.docs:
                score = 1.0 if doc_id in direct_doc_ids else 0.5
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _find_by_section(self, doc_type: str, section: str) -> list[str]:
        key = f"{doc_type}:{section}"
        return self.section_index.get(key, [])


# ─── KAG Merger ──────────────────────────────────────────

class KAGMerger:
    """
    3가지 검색 결과를 min-max 정규화 후 가중 합산.

    KAG 논문: alpha=0.5로 iterative blend.
    """

    def __init__(self, docs: dict = None, weights: dict[str, float] = None):
        self.docs = docs or {}
        self.weights = weights or {
            "kg_cs": 0.35,
            "kg_fr": 0.30,
            "rc": 0.15,
            "rc_embed": 0.20,
        }
        # GDC → doc_id 역인덱스
        self._gdc_to_docs: dict[str, set[str]] = defaultdict(set)
        # chapter → doc_id 역인덱스
        self._chapter_to_docs: dict[str, set[str]] = defaultdict(set)
        for doc_id, meta in self.docs.items():
            for gdc in meta.gdc_refs:
                self._gdc_to_docs[gdc].add(doc_id)
            chapter = meta.section_number.split(".")[0] if meta.section_number else ""
            if chapter:
                self._chapter_to_docs[chapter].add(doc_id)

    def merge(
        self,
        kg_cs_results: list[tuple[str, float]],
        kg_fr_results: list[tuple[str, float]],
        rc_results: list[tuple[str, float]],
        rc_embed_results: list[tuple[str, float]] = None,
        query: str = "",
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """4가지 결과 + concept intersection 보너스로 합산."""

        sources = {
            "kg_cs": kg_cs_results,
            "kg_fr": kg_fr_results,
            "rc": rc_results,
            "rc_embed": rc_embed_results or [],
        }

        # Adaptive weights: kg_cs가 없으면 kg_fr/rc_embed에 가중치 재분배
        weights = dict(self.weights)
        if not kg_cs_results:
            weights["kg_fr"] += weights["kg_cs"] * 0.5
            weights["rc_embed"] += weights["kg_cs"] * 0.3
            weights["rc"] += weights["kg_cs"] * 0.2
            weights["kg_cs"] = 0.0

        # min-max 정규화
        normalized = {}
        for name, results in sources.items():
            if not results:
                normalized[name] = {}
                continue
            scores = [s for _, s in results]
            min_s = min(scores)
            max_s = max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            normalized[name] = {
                doc_id: (score - min_s) / range_s
                for doc_id, score in results
            }

        # 가중 합산
        combined = defaultdict(float)
        source_count = defaultdict(int)

        for name, scores_dict in normalized.items():
            w = weights.get(name, 0.0)
            for doc_id, norm_score in scores_dict.items():
                combined[doc_id] += w * norm_score
                source_count[doc_id] += 1

        # 다수 소스에서 발견된 문서에 보너스
        for doc_id in combined:
            if source_count[doc_id] >= 2:
                combined[doc_id] *= 1.0 + 0.15 * (source_count[doc_id] - 1)

        # ── Concept Intersection Bonus ────────────────
        # 쿼리에서 GDC 주제 매칭 → 관련 GDC 추출
        # 쿼리에서 챕터 주제 매칭 → 관련 챕터 추출
        # GDC + 챕터 교집합 문서에 대폭 보너스
        if query and self.docs:
            matched_gdcs = self._match_gdc_concepts(query)
            matched_chapters = self._match_chapter_concepts(query)

            if matched_gdcs and matched_chapters:
                # 교집합: 매칭된 GDC를 인용하면서 매칭된 챕터에 속하는 문서
                gdc_docs = set()
                for gdc_num in matched_gdcs:
                    gdc_docs |= self._gdc_to_docs.get(gdc_num, set())
                chapter_docs = set()
                for ch in matched_chapters:
                    chapter_docs |= self._chapter_to_docs.get(ch, set())

                intersection = gdc_docs & chapter_docs
                if intersection:
                    # 교집합 문서에 강한 보너스
                    max_score = max(combined.values()) if combined else 1.0
                    for doc_id in intersection:
                        # 매칭된 GDC 수에 비례한 보너스
                        n_gdc_match = sum(
                            1 for g in matched_gdcs
                            if doc_id in self._gdc_to_docs.get(g, set())
                        )
                        combined[doc_id] = combined.get(doc_id, 0.0) + max_score * (0.8 + 0.2 * n_gdc_match)
            elif matched_gdcs:
                # GDC만 매칭: GDC 인용 문서에 보너스
                max_score = max(combined.values()) if combined else 1.0
                for gdc_num in matched_gdcs:
                    for doc_id in self._gdc_to_docs.get(gdc_num, set()):
                        combined[doc_id] = combined.get(doc_id, 0.0) + max_score * 0.3

        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _match_gdc_concepts(self, query: str) -> list[str]:
        """쿼리에서 GDC 주제와 매칭되는 GDC 번호를 반환."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        matched = []
        for gdc_num, desc in GDC_DESCRIPTIONS.items():
            desc_tokens = set(desc.split())
            overlap = query_tokens & desc_tokens
            # 최소 2개 주제 단어 매칭 (불용어 제외)
            meaningful = overlap - {"and", "of", "the", "for", "in", "to", "a", "an"}
            if len(meaningful) >= 2:
                matched.append(gdc_num)
        return matched

    def _match_chapter_concepts(self, query: str) -> list[str]:
        """쿼리에서 챕터 주제와 매칭되는 챕터 번호를 반환."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        matched = []
        for chapter, desc in DocBM25Retriever.CHAPTER_TOPICS.items():
            desc_tokens = set(desc.split())
            overlap = query_tokens & desc_tokens
            meaningful = overlap - {"and", "of", "the", "for", "in", "to", "a", "an"}
            if len(meaningful) >= 2:
                matched.append(chapter)
        return matched
