"""
LLM Query Decomposer (KAG LFPlanner-style)

자연어 쿼리를 구조화된 sub-queries + 예상 GDC/챕터로 분해.
암묵적 규제 참조를 명시적으로 추출하여 retriever 정확도 향상.
"""

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv("/home/bbo/RAG_Baseline/.env")


@dataclass
class DecomposedQuery:
    """분해된 쿼리 결과"""
    original: str
    sub_queries: list[str] = field(default_factory=list)
    likely_gdcs: list[str] = field(default_factory=list)
    likely_chapters: list[str] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are an NRC nuclear regulatory document retrieval specialist.

Your task: decompose a natural language query into structured retrieval hints for finding relevant NRC documents (SRP, DSRS, Regulatory Guides, 10 CFR).

You must output valid JSON with these fields:
- sub_queries: 2-4 reformulated search queries targeting different document types or regulatory concepts
- likely_gdcs: list of GDC numbers (strings) the query likely relates to
- likely_chapters: list of SRP/DSRS chapter numbers (strings like "2", "3", "6")
- key_concepts: 3-5 domain-specific terms for semantic search

GDC reference (10 CFR 50 Appendix A):
- GDC 2: natural phenomena (earthquakes, tornadoes, hurricanes, floods, tsunami)
- GDC 3: fire protection
- GDC 4: environmental/dynamic effects, missiles
- GDC 5: sharing of SSCs
- GDC 14-15: reactor coolant pressure boundary
- GDC 16: containment design
- GDC 17-18: electric power systems
- GDC 19: control room
- GDC 20-25: protection systems
- GDC 34: residual heat removal
- GDC 35-37: emergency core cooling
- GDC 38-40: containment heat removal
- GDC 41-43: containment atmosphere cleanup
- GDC 44-46: cooling water
- GDC 50-56: containment
- GDC 60-64: radioactive materials

SRP/DSRS chapters:
- Ch 2: site characteristics (meteorology, geology, hydrology, seismology)
- Ch 3: design of structures (seismic, wind, missiles, structural)
- Ch 4: reactor (core, thermal-hydraulic, fuel)
- Ch 5: reactor coolant system
- Ch 6: engineered safety features (containment, ECCS)
- Ch 7: instrumentation and controls
- Ch 8: electric power
- Ch 9: auxiliary systems
- Ch 10: steam and power conversion
- Ch 11: radioactive waste management
- Ch 12: radiation protection
- Ch 13: conduct of operations
- Ch 14: initial test program
- Ch 15: transient and accident analyses"""

FEW_SHOT = """Example 1:
Query: "What are the acceptance criteria for containment heat removal systems?"
Output: {"sub_queries": ["GDC 38 containment heat removal acceptance criteria", "SRP 6.2.2 containment heat removal systems review", "containment cooling post-accident pressure temperature reduction"], "likely_gdcs": ["38", "39", "40", "50"], "likely_chapters": ["6"], "key_concepts": ["containment heat removal", "post-accident cooling", "pressure reduction", "ultimate heat sink"]}

Example 2:
Query: "How do regulatory requirements for natural phenomena define the geographic scope for site parameters?"
Output: {"sub_queries": ["GDC 2 natural phenomena design bases site", "SRP 2.3 regional climatology meteorological parameters", "site characteristics geographic scope natural hazards assessment"], "likely_gdcs": ["2", "4"], "likely_chapters": ["2"], "key_concepts": ["natural phenomena", "site parameters", "geographic scope", "regional climatology", "design basis"]}

Example 3:
Query: "What seismic design parameters must be established for nuclear structures?"
Output: {"sub_queries": ["GDC 2 seismic design bases earthquake protection", "SRP 3.7.1 seismic design parameters SSE OBE", "DSRS 3.7.1 seismic design parameters NuScale"], "likely_gdcs": ["2", "4"], "likely_chapters": ["2", "3"], "key_concepts": ["seismic design parameters", "safe shutdown earthquake", "seismic Category I", "ground motion"]}"""


class QueryDecomposer:
    """GPT-4.1 기반 쿼리 분해."""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return None
                self._client = OpenAI(api_key=api_key)
            except Exception:
                return None
        return self._client

    def decompose(self, query: str) -> DecomposedQuery:
        """쿼리를 sub-queries + GDC/chapter hints로 분해."""
        client = self._get_client()
        if client is None:
            return DecomposedQuery(original=query)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": FEW_SHOT + f"\n\nQuery: \"{query}\"\nOutput:"},
                ],
                max_tokens=500,
                temperature=0.0,
                timeout=5,
            )
            raw = response.choices[0].message.content.strip()
            # JSON 추출
            cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
            data = json.loads(cleaned)

            return DecomposedQuery(
                original=query,
                sub_queries=data.get("sub_queries", [])[:4],
                likely_gdcs=[str(g) for g in data.get("likely_gdcs", [])][:6],
                likely_chapters=[str(c) for c in data.get("likely_chapters", [])][:4],
                key_concepts=data.get("key_concepts", [])[:5],
            )
        except Exception:
            return DecomposedQuery(original=query)
