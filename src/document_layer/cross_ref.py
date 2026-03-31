"""
Cross-Reference Extractor

각 문서에서 다른 문서에 대한 참조를 regex로 추출한다.
- 10 CFR 참조 (e.g., "10 CFR 50.46")
- GDC 참조 (e.g., "GDC 38")
- RG 참조 (e.g., "Regulatory Guide 1.82", "RG 1.82")
- SRP 참조 (e.g., "SRP 6.2.2", "NUREG-0800 Section 6.2.2")
- DSRS 참조 (e.g., "DSRS 6.2.1")
"""

import re
import os
from pathlib import Path

import fitz  # PyMuPDF

from src.document_layer.classifier import DocumentMeta


# ─── 참조 패턴 ─────────────────────────────────────────
PATTERNS = {
    "cfr": re.compile(
        r'10\s*CFR\s*(?:Part\s*)?([\d]+(?:\.[\d]+[a-z]*(?:\([\w]+\))*)*)',
        re.IGNORECASE,
    ),
    "gdc": re.compile(
        r'GDC\s+(\d+)',
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


def extract_cross_refs(doc: DocumentMeta) -> DocumentMeta:
    """단일 문서에서 cross-reference를 추출하여 메타데이터에 채운다."""
    pdf_path = doc.pdf_path
    if not pdf_path or not os.path.isfile(pdf_path):
        return doc

    try:
        pdf = fitz.open(pdf_path)
        full_text = ""
        for page in pdf:
            full_text += page.get_text()
        pdf.close()
    except Exception as e:
        print(f"[WARN] Cannot read {doc.filename}: {e}")
        return doc

    doc.cfr_refs = sorted(set(PATTERNS["cfr"].findall(full_text)))
    doc.gdc_refs = sorted(set(PATTERNS["gdc"].findall(full_text)))
    doc.rg_refs = sorted(set(PATTERNS["rg"].findall(full_text)))
    doc.srp_refs = sorted(set(PATTERNS["srp"].findall(full_text)))
    doc.dsrs_refs = sorted(set(PATTERNS["dsrs"].findall(full_text)))

    # 자기 자신 참조 제거
    if doc.doc_type == "RG" and doc.section_number in doc.rg_refs:
        doc.rg_refs.remove(doc.section_number)
    if doc.doc_type == "SRP" and doc.section_number in doc.srp_refs:
        doc.srp_refs.remove(doc.section_number)
    if doc.doc_type == "DSRS" and doc.section_number in doc.dsrs_refs:
        doc.dsrs_refs.remove(doc.section_number)

    return doc


def extract_all_cross_refs(docs: list[DocumentMeta]) -> list[DocumentMeta]:
    """모든 문서에서 cross-reference를 추출한다."""
    for i, doc in enumerate(docs):
        extract_cross_refs(doc)
        total_refs = (
            len(doc.cfr_refs) + len(doc.gdc_refs) + len(doc.rg_refs)
            + len(doc.srp_refs) + len(doc.dsrs_refs)
        )
        if (i + 1) % 20 == 0 or i == len(docs) - 1:
            print(f"[{i+1}/{len(docs)}] {doc.display_name}: {total_refs} refs")

    return docs
