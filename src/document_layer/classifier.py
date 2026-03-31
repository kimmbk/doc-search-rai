"""
Document Classifier & Metadata Extractor

151개 NRC 규제문서를 4가지 유형으로 분류하고 메타데이터를 추출한다.
- REG: 연방규정 (10 CFR Part 50)
- RG: Regulatory Guide
- SRP: Standard Review Plan (NUREG-0800)
- DSRS: Design-Specific Review Standard (NuScale SMR)
"""

import re
import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class DocumentMeta:
    """문서 메타데이터"""
    doc_id: str              # ML번호 또는 파일명 기반 ID
    filename: str
    doc_type: str            # REG, RG, SRP, DSRS
    section_number: str      # e.g., "50.46", "1.82", "6.2.2"
    title: str
    revision: str = ""
    page_count: int = 0
    pdf_path: str = ""
    issue_date: str = ""
    # cross-references (추후 추출)
    cfr_refs: list = field(default_factory=list)
    gdc_refs: list = field(default_factory=list)
    rg_refs: list = field(default_factory=list)
    srp_refs: list = field(default_factory=list)
    dsrs_refs: list = field(default_factory=list)
    section_refs: list = field(default_factory=list)

    @property
    def display_name(self) -> str:
        prefix = {"REG": "10 CFR", "RG": "RG", "SRP": "SRP", "DSRS": "DSRS"}
        return f"{prefix.get(self.doc_type, '')} {self.section_number}"

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentClassifier:
    """PDF 문서를 분류하고 메타데이터를 추출한다."""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)

    def classify_all(self) -> list[DocumentMeta]:
        """디렉토리 내 모든 PDF를 분류한다."""
        results = []
        pdf_files = sorted(self.docs_dir.glob("*.pdf"))

        for pdf_path in pdf_files:
            meta = self._classify_single(pdf_path)
            if meta:
                results.append(meta)

        return results

    def _classify_single(self, pdf_path: Path) -> Optional[DocumentMeta]:
        """단일 PDF를 분류하고 메타데이터를 추출한다."""
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            for i in range(min(3, doc.page_count)):
                text += doc[i].get_text()

            page_count = doc.page_count
            doc.close()
        except Exception as e:
            print(f"[WARN] Cannot read {pdf_path.name}: {e}")
            return None

        filename = pdf_path.name
        doc_id = filename.replace(".pdf", "").replace(" ", "_")

        # 분류 우선순위: REG > DSRS > SRP > RG
        if "10 CFR" in filename or "10 CFR" in text[:100]:
            return self._extract_reg(doc_id, filename, text, page_count, str(pdf_path))

        if re.search(r'DESIGN[- ]*SPECIFIC REVIEW STANDARD', text) or "DESIGN SPECIFIC REVIEW STANDARD" in text:
            return self._extract_dsrs(doc_id, filename, text, page_count, str(pdf_path))

        if "STANDARD REVIEW PLAN" in text:
            return self._extract_srp(doc_id, filename, text, page_count, str(pdf_path))

        if "REGULATORY GUIDE" in text:
            return self._extract_rg(doc_id, filename, text, page_count, str(pdf_path))

        # 분류 실패
        print(f"[WARN] Unclassified: {filename}")
        return DocumentMeta(
            doc_id=doc_id, filename=filename, doc_type="UNKNOWN",
            section_number="", title="", page_count=page_count,
            pdf_path=str(pdf_path),
        )

    # ─── REG ─────────────────────────────────────────────
    def _extract_reg(self, doc_id, filename, text, page_count, pdf_path) -> DocumentMeta:
        m = re.search(r'Part\s+(\d+)', text)
        section = m.group(1) if m else "50"

        return DocumentMeta(
            doc_id=doc_id, filename=filename, doc_type="REG",
            section_number=section,
            title="Domestic Licensing of Production and Utilization Facilities",
            page_count=page_count, pdf_path=pdf_path,
        )

    # ─── DSRS ────────────────────────────────────────────
    def _extract_dsrs(self, doc_id, filename, text, page_count, pdf_path) -> DocumentMeta:
        section = ""
        title = ""

        # 섹션번호: "for NuScale" 뒤에 나오는 첫 번호 + 대문자 제목
        m = re.search(
            r'(?:DESIGN[- ]*SPECIFIC REVIEW STANDARD|DESIGN SPECIFIC REVIEW STANDARD)'
            r'.*?for NuScale.*?\n\s*([\d]+\.[\d]+(?:\.[\d]+)*)\s*(.*?)(?:\n|REVIEW)',
            text, re.DOTALL
        )
        if m:
            section = m.group(1)
            title = m.group(2).strip()
        else:
            # fallback: 첫 번째 숫자.숫자 패턴 + 대문자 제목
            m = re.search(
                r'\n\s*([\d]{1,2}\.[\d]{1,2}(?:\.[\d]{1,2})?)\s+\n?\s*([A-Z][A-Z\s\-\(\)/,]+)',
                text
            )
            if m:
                section = m.group(1)
                title = m.group(2).strip()

        # revision
        rev = ""
        m_rev = re.search(r'Revision\s+([\d]+)', text[:500])
        if m_rev:
            rev = m_rev.group(1)

        title = re.sub(r'\s+', ' ', title).strip().rstrip(',').rstrip()

        return DocumentMeta(
            doc_id=doc_id, filename=filename, doc_type="DSRS",
            section_number=section, title=title, revision=rev,
            page_count=page_count, pdf_path=pdf_path,
        )

    # ─── SRP ─────────────────────────────────────────────
    def _extract_srp(self, doc_id, filename, text, page_count, pdf_path) -> DocumentMeta:
        section = ""
        title = ""

        # 패턴: 숫자.숫자 + 대문자 제목 (NUREG-0800 보일러플레이트 이후)
        # STANDARD REVIEW PLAN 보일러플레이트를 건너뛴 후 첫 번호
        # 보일러플레이트 끝: "...identifies differences..." 또는 "REACTOR REGULATION"
        boilerplate_end = 0
        for marker in ["REACTOR REGULATION", "identifies differences", "However, an applicant"]:
            idx = text.find(marker)
            if idx > 0:
                boilerplate_end = max(boilerplate_end, idx)

        search_text = text[boilerplate_end:] if boilerplate_end > 0 else text

        m = re.search(
            r'([\d]{1,2}\.[\d]{1,2}(?:\.[\d]{1,2})?)\s+([A-Z][A-Z\s\-\(\)/,\:]+)',
            search_text
        )
        if m:
            section = m.group(1)
            title = m.group(2).strip()

        # revision
        rev = ""
        m_rev = re.search(r'Revision\s+([\d]+)', text[:500])
        if m_rev:
            rev = m_rev.group(1)

        title = re.sub(r'\s+', ' ', title).strip()
        # 제목 끝에 불필요한 부분 잘라내기
        for cutoff in ["REVIEW RESPONSIBILITIES", "APPENDIX", "\n"]:
            if cutoff in title:
                title = title[:title.index(cutoff)].strip()

        return DocumentMeta(
            doc_id=doc_id, filename=filename, doc_type="SRP",
            section_number=section, title=title, revision=rev,
            page_count=page_count, pdf_path=pdf_path,
        )

    # ─── RG ──────────────────────────────────────────────
    def _extract_rg(self, doc_id, filename, text, page_count, pdf_path) -> DocumentMeta:
        rg_num = ""
        title = ""

        # RG 번호 추출
        m = re.search(r'REGULATORY GUIDE\s+(?:RG\s+)?([\d]+\.[\d]+)', text)
        if m:
            rg_num = m.group(1)

        # 제목: RG 번호 바로 뒤 또는 "(Task..." 뒤
        # 패턴: 대문자로 된 제목
        if rg_num:
            # RG 번호 뒤에서 제목 찾기
            pattern = rf'REGULATORY GUIDE\s+(?:RG\s+)?{re.escape(rg_num)}.*?\n\s*(.+?)(?:\n\s*A\.\s|$)'
            m_title = re.search(pattern, text, re.DOTALL)
            if m_title:
                raw = m_title.group(1).strip()
                # 여러 줄일 수 있으므로 첫 의미있는 줄
                for line in raw.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('(') and len(line) > 5:
                        title = line
                        break

        # revision
        rev = ""
        m_rev = re.search(r'Revision\s+([\d]+)', text[:500])
        if m_rev:
            rev = m_rev.group(1)

        # issue date
        issue_date = ""
        m_date = re.search(r'Issue Date:\s*(\w+\s+\d{4})', text[:1000])
        if m_date:
            issue_date = m_date.group(1)

        title = re.sub(r'\s+', ' ', title).strip()

        return DocumentMeta(
            doc_id=doc_id, filename=filename, doc_type="RG",
            section_number=rg_num, title=title, revision=rev,
            page_count=page_count, pdf_path=pdf_path,
            issue_date=issue_date,
        )


def save_metadata(docs: list[DocumentMeta], output_path: str) -> None:
    """메타데이터를 JSON으로 저장한다."""
    data = [d.to_dict() for d in docs]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(data)} documents to {output_path}")


def load_metadata(input_path: str) -> list[DocumentMeta]:
    """저장된 메타데이터를 로드한다."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs = []
    for d in data:
        # dataclass 필드만 추출
        meta = DocumentMeta(
            doc_id=d["doc_id"], filename=d["filename"],
            doc_type=d["doc_type"], section_number=d["section_number"],
            title=d["title"], revision=d.get("revision", ""),
            page_count=d.get("page_count", 0), pdf_path=d.get("pdf_path", ""),
            issue_date=d.get("issue_date", ""),
            cfr_refs=d.get("cfr_refs", []),
            gdc_refs=d.get("gdc_refs", []),
            rg_refs=d.get("rg_refs", []),
            srp_refs=d.get("srp_refs", []),
            dsrs_refs=d.get("dsrs_refs", []),
            section_refs=d.get("section_refs", []),
        )
        docs.append(meta)
    return docs
