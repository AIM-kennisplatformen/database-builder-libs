from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pypdf import PdfReader

from docling_core.types.doc import SectionHeaderItem, TextItem

from database_builder_libs.models.abstract_chunk_embedder import AbstractChunkEmbedder
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy
from database_builder_libs.models.abstract_source import AbstractSource, Content
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.chunk.n_points_section import SectionChunkingStrategy
from database_builder_libs.utility.extract.document_parser_docling import (
    DocumentConversionError,
    DocumentParserDocling,
    ParsedDocument,
)


@dataclasses.dataclass(slots=True)
class Institution:
    name: str
    parent: Optional[str] = None


@dataclasses.dataclass(slots=True)
class Acknowledgement:
    name: str
    type: str      # "person" | "organization" | "group"
    relation: str  # "funding" | "collaboration" | "contribution" | "review" | "support"


@dataclasses.dataclass(slots=True)
class DocumentMetadata:
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publishing_institute: Optional[Institution] = None
    summary: Optional[str] = None
    acknowledgements: List[Acknowledgement] = dataclasses.field(default_factory=list)
    source: Dict[str, str] = dataclasses.field(default_factory=dict)
    keywords: Optional[List[str]] = None
    literature_type: Optional[str] = None
    strategic_overview: Optional[List[str]] = None
    target_groups: Optional[List[str]] = None
    best_practices: Optional[List[str]] = None

class ExtractionStrategy(str, Enum):
    FILE_METADATA = "file_metadata"
    DOCLING       = "docling"
    LLM           = "llm"


class OrderedStrategyConfig(BaseModel):
    order: List[ExtractionStrategy] = Field(default_factory=list)
    stop_on_success: bool = True

    @field_validator("order")
    def no_duplicates(cls, v: list) -> list:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate strategies in order")
        return v


class FieldExtractionConfig(BaseModel):
    enabled: bool = True
    strategies: OrderedStrategyConfig = Field(default_factory=OrderedStrategyConfig)


class SectionsConfig(BaseModel):
    enabled: bool = True
    chunking_strategy: AbstractChunkingStrategy = Field(
        default_factory=SectionChunkingStrategy
    )
    embedder: Optional[AbstractChunkEmbedder] = None
    model_config = {"arbitrary_types_allowed": True}


class PDFDocumentConfig(BaseModel):
    """
    Full configuration for the PDFSource pipeline.

    Every metadata field has a ``FieldExtractionConfig`` with an ordered list
    of :class:`ExtractionStrategy` values.  The pipeline tries each strategy
    in order and stops on first success when ``stop_on_success=True``.

    Available strategies
    --------------------
    ``FILE_METADATA``
        Reads embedded PDF metadata via ``pypdf`` (fast, no ML).
    ``DOCLING``
        Infers fields from the Docling structural IR.
    ``LLM``
        Sends the first ~60 lines to the configured LLM.
        Requires ``llm_base_url`` and ``llm_api_key`` to be set.

    Defaults
    --------
    - ``title``               : FILE_METADATA → DOCLING
    - ``authors``             : FILE_METADATA → LLM
    - ``summary``             : DOCLING
    - ``publishing_institute``: FILE_METADATA
    - ``acknowledgements``    : LLM
    """

    folder_path: Path

    title: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.DOCLING]
            )
        )
    )
    authors: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.LLM]
            )
        )
    )
    summary: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(order=[ExtractionStrategy.DOCLING])
        )
    )
    publishing_institute: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(order=[ExtractionStrategy.FILE_METADATA])
        )
    )
    acknowledgements: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
        )
    )

    sections: SectionsConfig = Field(default_factory=SectionsConfig)

    llm_base_url: Optional[str] = None
    llm_api_key:  Optional[str] = None
    llm_model:    str = "gpt-4.1-mini"


_SUMMARY_HEADERS   = frozenset({"abstract", "summary", "executive summary", "samenvatting"})
_PDF_META_JUNK     = frozenset({"unknown", "untitled", "microsoft word", "writer", "author"})
_INSTITUTION_HINTS = frozenset({
    "university", "institute", "research", "centre", "center", "group",
    "foundation", "association", "agency", "organisation", "organization",
    "network", "ministry", "department",
})


class PDFSource(AbstractSource):
    """
    Self-contained PDF source that parses, extracts metadata, chunks, and
    embeds in a single configurable pipeline.

    Mapping
    -------
    PDF file path (relative)  →  Content.id_
    File mtime (UTC)          →  Content.date

    Content.content keys
    --------------------
    ``file_path``, ``file_name``, ``file_size``, ``num_pages``
        File-level stats.
    ``pdf_meta``
        Raw pypdf embedded metadata dict.
    ``num_sections``, ``num_tables``, ``num_figures``, ``section_titles``
        Structural counts from Docling.
    ``metadata``
        :class:`DocumentMetadata` serialised via ``dataclasses.asdict()``.
    ``chunks``
        List of :class:`~database_builder_libs.models.chunk.Chunk` dicts.
    """

    _config:     Optional[PDFDocumentConfig]    = PrivateAttr(default=None)
    _parser:     Optional[DocumentParserDocling] = PrivateAttr(default=None)
    _llm_client: Any                             = PrivateAttr(default=None)

    def _connect_impl(self, config: Mapping[str, Any]) -> None:
        self._config = PDFDocumentConfig(**config)
        if not self._config.folder_path.is_dir():
            raise ValueError(
                f"folder_path '{self._config.folder_path}' does not exist or is not a directory."
            )
        self._parser     = DocumentParserDocling()
        self._llm_client = self._build_llm_client()
        logger.info(f"PDFSource connected to folder: {self._config.folder_path}")

    def get_all_documents_metadata(self, limit: int = -1) -> List[dict[str, Any]]:
        """Return lightweight file-level metadata for all PDFs (no Docling conversion)."""
        self._ensure_connected()
        assert self._config is not None
        results: List[dict[str, Any]] = []
        for pdf_path in sorted(self._config.folder_path.rglob("*.pdf")):
            if limit != -1 and len(results) >= limit:
                break
            stat = pdf_path.stat()
            results.append({
                "id":       str(pdf_path.relative_to(self._config.folder_path)),
                "path":     pdf_path,
                "size":     stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                "pdf_meta": self._read_pdf_meta(str(pdf_path)),
            })
        return results

    def get_list_artefacts(self, last_synced: Optional[datetime]) -> list[tuple[str, datetime]]:
        """Return PDFs modified after ``last_synced``, sorted ascending by mtime."""
        self._ensure_connected()
        assert self._config is not None
        if last_synced is not None and last_synced.tzinfo is None:
            last_synced = last_synced.replace(tzinfo=timezone.utc)
        artefacts = [
            (str(p.relative_to(self._config.folder_path)),
             datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc))
            for p in self._config.folder_path.rglob("*.pdf")
            if last_synced is None
            or datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) > last_synced
        ]
        artefacts.sort(key=lambda t: t[1])
        logger.info(f"get_list_artefacts: {len(artefacts)} PDF(s) since {last_synced}.")
        return artefacts

    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """Run the full parse → metadata → chunk → embed pipeline for each artefact."""
        self._ensure_connected()
        assert self._config is not None
        assert self._parser is not None

        contents: list[Content] = []
        for relative_id, modified in artefacts:
            pdf_path = self._config.folder_path / relative_id
            if not pdf_path.exists():
                raise KeyError(f"Artefact '{relative_id}' no longer exists.")

            stat         = pdf_path.stat()
            pdf_path_str = str(pdf_path.resolve())

            parsed: Optional[ParsedDocument] = None
            num_pages: Optional[int] = None
            try:
                parsed    = self._parser.parse(pdf_path_str)
                num_pages = len(parsed.doc.pages) if parsed.doc.pages else None
            except DocumentConversionError as exc:
                logger.warning(f"Docling conversion failed for '{relative_id}': {exc}")
            except Exception as exc:
                logger.warning(f"Unexpected parse error for '{relative_id}': {exc}")

            metadata = self._extract_metadata(pdf_path_str=pdf_path_str, parsed=parsed)
            chunks   = self._chunk(parsed=parsed, document_id=relative_id) if (
                self._config.sections.enabled and parsed is not None
            ) else []
            if chunks and self._config.sections.embedder is not None:
                chunks = self._embed(chunks)

            contents.append(Content(
                date=modified,
                id_=relative_id,
                content={
                    "file_path":  pdf_path_str,
                    "file_name":  pdf_path.name,
                    "file_size":  stat.st_size,
                    "num_pages":  num_pages,
                    "pdf_meta":   self._read_pdf_meta(pdf_path_str),
                    "metadata":   dataclasses.asdict(metadata),
                    **self._structural_meta(parsed),
                    "chunks":     [dataclasses.asdict(c) for c in chunks],
                },
            ))
            logger.debug(f"Processed '{relative_id}': {len(chunks)} chunk(s).")

        logger.info(f"get_content returning {len(contents)} Content object(s).")
        return contents

    def _extract_metadata(
        self, *, pdf_path_str: str, parsed: Optional[ParsedDocument]
    ) -> DocumentMetadata:
        """
        Populate DocumentMetadata by running each field's configured strategy
        order, stopping on first success when stop_on_success=True.
        """
        assert self._config is not None
        meta = DocumentMetadata()
        if parsed is None:
            return meta

        try:
            # ── lazy, cached accessors ────────────────────────────────────
            _cache: dict[str, Any] = {}

            def lines() -> List[str]:
                if "lines" not in _cache:
                    _cache["lines"] = self._first_lines(parsed.doc, limit=120)
                return _cache["lines"]

            def pdf_info() -> dict[str, Any]:
                if "pdf_info" not in _cache:
                    try:
                        raw = PdfReader(pdf_path_str).metadata or {}
                        # raw may be a pypdf DocumentInformation object rather than
                        # a plain dict — iterate items() but only keep entries whose
                        # keys are real strings so MagicMock / PrivateAttr objects
                        # from mismatched pydantic versions are silently dropped.
                        info: dict[str, Any] = {}
                        for k, v in raw.items():
                            if isinstance(k, str):
                                info[k.lstrip("/")] = v
                        # Also check well-known attributes directly in case the
                        # object exposes them as properties (pypdf DocumentInformation).
                        for attr, key in (("title", "Title"), ("author", "Author"),
                                          ("producer", "Producer"), ("creator", "Creator")):
                            if key not in info:
                                val = getattr(raw, attr, None)
                                if isinstance(val, str):
                                    info[key] = val
                        _cache["pdf_info"] = info
                    except Exception:
                        _cache["pdf_info"] = {}
                return _cache["pdf_info"]

            def llm() -> dict[str, Any]:
                if "llm" not in _cache:
                    _cache["llm"] = self._call_llm(lines()) if self._llm_client else {}
                return _cache["llm"]

            # ── per-field extractors: return (value, source_label) or None ─
            S = ExtractionStrategy

            def _title(s: S) -> Optional[Tuple[Any, str]]:
                if s == S.FILE_METADATA:
                    v = self._clean_meta_string(pdf_info().get("Title"))
                    return (v, "pdf_metadata") if v else None
                if s == S.DOCLING:
                    v = self._first_section_header(parsed.doc) or self._first_reasonable_line(lines())
                    return (v, "docling_heuristic") if v else None
                if s == S.LLM and self._llm_client:
                    v = llm().get("title")
                    return (v, "llm") if v else None

            def _authors(s: S) -> Optional[Tuple[Any, str]]:
                if s == S.FILE_METADATA:
                    raw = self._clean_meta_string(pdf_info().get("Author"))
                    return (self._split_authors(raw), "pdf_metadata") if raw else None
                if s == S.DOCLING:
                    for line in lines()[:10]:
                        a = self._parse_author_line(line)
                        if len(a) >= 2:
                            return (a, "docling_heuristic")
                if s == S.LLM and self._llm_client:
                    v = llm().get("authors")
                    return (v, "llm") if v else None

            def _summary(s: S) -> Optional[Tuple[Any, str]]:
                if s == S.DOCLING:
                    v = self._find_summary(parsed.doc)
                    return (v, "docling_heuristic") if v else None
                if s == S.LLM and self._llm_client:
                    v = llm().get("summary")
                    return (v, "llm") if v else None

            def _institute(s: S) -> Optional[Tuple[Any, str]]:
                if s == S.FILE_METADATA:
                    raw = self._clean_meta_string(
                        pdf_info().get("Producer") or pdf_info().get("Creator")
                    )
                    if raw and not any(j in raw.lower() for j in _PDF_META_JUNK):
                        return (Institution(name=raw), "pdf_metadata")
                if s == S.LLM and self._llm_client:
                    v = llm().get("publishing_institute")
                    return (Institution(name=v), "llm") if v else None

            def _acks(s: S) -> Optional[Tuple[Any, str]]:
                if s == S.LLM and self._llm_client:
                    items = [
                        Acknowledgement(
                            name=a["name"],
                            type=a.get("type", ""),
                            relation=a.get("relation", ""),
                        )
                        for a in llm().get("acknowledgements", [])
                        if a.get("name")
                    ]
                    return (items, "llm") if items else None

            # ── run each field through its strategy cascade ───────────────
            fields = [
                (self._config.title,               _title,    "title"),
                (self._config.authors,             _authors,  "authors"),
                (self._config.summary,             _summary,  "summary"),
                (self._config.publishing_institute, _institute, "publishing_institute"),
                (self._config.acknowledgements,    _acks,     "acknowledgements"),
            ]

            for cfg, extractor, attr in fields:
                if not cfg.enabled:
                    continue
                for strategy in cfg.strategies.order:
                    result = extractor(strategy)
                    if result is not None:
                        value, source_label = result
                        setattr(meta, attr, value)
                        meta.source[attr] = source_label
                        if cfg.strategies.stop_on_success:
                            break

        except Exception as exc:
            logger.warning(f"Metadata extraction failed for '{pdf_path_str}': {exc}")

        return meta

    def _build_llm_client(self) -> Optional[Any]:
        assert self._config is not None
        if not self._config.llm_base_url or not self._config.llm_api_key:
            return None
        try:
            from openai import OpenAI
            return OpenAI(base_url=self._config.llm_base_url, api_key=self._config.llm_api_key)
        except ImportError:
            logger.warning("openai package not installed; LLM extraction skipped.")
            return None

    def _call_llm(self, lines: List[str]) -> dict[str, Any]:
        assert self._config is not None
        prompt = f"""Extract metadata from the document header. Return only JSON.

{{
  "title": "Full document title or null",
  "authors": ["First Last"],
  "publishing_institute": "Name of publisher or null",
  "acknowledgements": [
    {{
      "name": "Entity name",
      "type": "person | organization | group",
      "relation": "funding | collaboration | contribution | review | support"
    }}
  ]
}}

Rules:
- title is the actual paper/report title, not a repository name, journal name, or tool artefact.
- authors must be personal names only; ignore job titles and copyright text.
- publishing_institute is the organisation that published or issued the document.
- Extract acknowledgement entities mentioned in the header or acknowledgements section.

Text:
{chr(10).join(lines[:60])}
"""
        try:
            res     = self._llm_client.chat.completions.create(
                model=self._config.llm_model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = res.choices[0].message.content.strip()
            match   = re.search(r"\{.*\}", content, re.S)
            return json.loads(match.group(0)) if match else {}
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")
            return {}

    @staticmethod
    def _first_lines(doc: Any, limit: int) -> List[str]:
        out: List[str] = []
        for node, _ in doc.iterate_items():
            if isinstance(node, (SectionHeaderItem, TextItem)):
                for ln in (node.text or "").strip().splitlines():
                    if ln := ln.strip():
                        out.append(ln)
                        if len(out) >= limit:
                            return out
        return out

    _TITLE_NOISE = frozenset({
        "university of huddersfield repository", "original citation", "repository",
        "preprint", "author's accepted manuscript", "accepted manuscript", "post-print",
    })

    @classmethod
    def _looks_like_title(cls, t: str) -> bool:
        if len(t) < 8 or "@" in t or len(t.split()) < 2:
            return False
        if t == t.upper() and len(t) > 30:
            return False
        return t.strip().lower() not in cls._TITLE_NOISE

    @classmethod
    def _first_section_header(cls, doc: Any) -> Optional[str]:
        for node, _ in doc.iterate_items():
            if isinstance(node, SectionHeaderItem):
                if t := (node.text or "").strip():
                    if cls._looks_like_title(t):
                        return t
        return None

    @classmethod
    def _first_reasonable_line(cls, lines: List[str]) -> Optional[str]:
        return next((ln for ln in lines[:30] if cls._looks_like_title(ln)), None)

    @staticmethod
    def _find_summary(doc: Any) -> Optional[str]:
        collecting, collected = False, []
        for node, _ in doc.iterate_items():
            if isinstance(node, SectionHeaderItem):
                h = (node.text or "").strip().lower()
                if any(k in h for k in _SUMMARY_HEADERS):
                    collecting = True
                    continue
                if collecting:
                    break
            if collecting and isinstance(node, TextItem):
                if t := (node.text or "").strip():
                    collected.append(t)
        return "\n".join(collected[:4]) or None

    @staticmethod
    def _read_pdf_meta(pdf_path_str: str) -> dict[str, Any]:
        try:
            if meta := PdfReader(pdf_path_str).metadata:
                return {k.lstrip("/"): v for k, v in meta.items() if v is not None}
        except Exception as exc:
            logger.warning(f"Could not read PDF metadata for '{pdf_path_str}': {exc}")
        return {}

    @staticmethod
    def _clean_meta_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in _PDF_META_JUNK or s.lower().startswith("microsoft word"):
            return None
        return s

    @staticmethod
    def _split_authors(s: str) -> List[str]:
        if ";" in s:
            parts = [p.strip() for p in s.split(";")]
        elif re.search(r"\band\b", s, flags=re.I):
            parts = [p.strip() for p in re.split(r"\band\b", s, flags=re.I)]
        else:
            parts = [s.strip()]
        return [p for p in parts if p]

    @staticmethod
    def _parse_author_line(text: str) -> List[str]:
        """Parse abbreviated author lines like ``Smith J., Doe A.``"""
        pattern = re.compile(r"^([A-Z][a-zA-Z\-']+)\s+([A-Z])\.?$")
        authors = []
        for part in text.replace(" and ", ",").split(","):
            if m := pattern.match(part.strip()):
                last, initial = m.groups()
                authors.append(f"{initial} {last}")
        return authors

    def _chunk(self, *, parsed: ParsedDocument, document_id: str) -> list[Chunk]:
        assert self._config is not None
        try:
            chunks = self._config.sections.chunking_strategy.chunk(
                parsed.sections,
                document_id=document_id,
                summary=self._find_summary(parsed.doc),
            )
            logger.debug(f"Chunked '{document_id}': {len(chunks)} chunk(s).")
            return chunks
        except Exception as exc:
            logger.warning(f"Chunking failed for '{document_id}': {exc}")
            return []

    def _embed(self, chunks: list[Chunk]) -> list[Chunk]:
        assert self._config is not None
        assert self._config.sections.embedder is not None
        try:
            embedded = self._config.sections.embedder.embed(chunks)
            logger.debug(f"Embedded {len(embedded)} chunk(s).")
            return embedded
        except Exception as exc:
            logger.warning(f"Embedding failed; returning chunks without vectors: {exc}")
            return chunks

    @staticmethod
    def _structural_meta(parsed: Optional[ParsedDocument]) -> dict[str, Any]:
        if parsed is None:
            return {"num_sections": 0, "num_tables": 0, "num_figures": 0, "section_titles": []}
        return {
            "num_sections":   len(parsed.sections),
            "num_tables":     len(parsed.tables),
            "num_figures":    len(parsed.figures),
            "section_titles": [t for t, _, _ in parsed.sections if t],
        }