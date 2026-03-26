from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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


# --------------------------------------------------------------------------- #
# Extracted metadata                                                           #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(slots=True)
class Institution:
    """A publishing institution."""
    name: str
    parent: Optional[str] = None


@dataclasses.dataclass(slots=True)
class Acknowledgement:
    """An entity acknowledged in the document."""
    name: str
    type: str      # "person" | "organization" | "group"
    relation: str  # "funding" | "collaboration" | "contribution" | "review" | "support"


@dataclasses.dataclass(slots=True)
class DocumentMetadata:
    """
    Rich metadata extracted from a PDF document.

    Produced by :class:`PDFSource` and serialised into
    ``Content.content["metadata"]`` via ``dataclasses.asdict()``.

    Fields
    ------
    title : str | None
        Document title.
    authors : list[str] | None
        Personal author names.
    publishing_institute : Institution | None
        Publisher or issuing organisation.
    summary : str | None
        Abstract or executive summary text.
    acknowledgements : list[Acknowledgement]
        Entities acknowledged in the document.
    source : dict[str, str]
        Maps each populated field to the strategy that filled it
        (e.g. ``{"title": "docling_heuristic", "authors": "llm"}``).
    keywords : list[str] | None
        Keywords or tags associated with the document.
    literature_type : str | None
        Document type (e.g. ``"report"``, ``"article"``).
    strategic_overview : list[str] | None
        Strategic themes extracted from tags.
    target_groups : list[str] | None
        Target audience groups extracted from tags.
    best_practices : list[str] | None
        Best practice topics extracted from tags.
    """

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


# --------------------------------------------------------------------------- #
# Strategy config                                                              #
# --------------------------------------------------------------------------- #

class ExtractionStrategy(str, Enum):
    """Which engine to use when extracting a metadata field."""
    FILE_METADATA = "file_metadata"  # embedded PDF metadata via pypdf
    DOCLING       = "docling"        # structural heuristics from Docling IR
    LLM           = "llm"            # LLM extraction from document header text


class OrderedStrategyConfig(BaseModel):
    """Ordered list of strategies to try for a single metadata field."""

    order: List[ExtractionStrategy] = Field(default_factory=list)
    stop_on_success: bool = True

    @field_validator("order")
    def no_duplicates(cls, v: list) -> list:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate strategies in order")
        return v


class FieldExtractionConfig(BaseModel):
    """Configuration for extracting a single metadata field."""

    enabled: bool = True
    strategies: OrderedStrategyConfig = Field(default_factory=OrderedStrategyConfig)


class SectionsConfig(BaseModel):
    """
    Controls section extraction, chunking, and optional embedding.

    Attributes
    ----------
    enabled : bool
        When ``False`` no chunks are produced.
    chunking_strategy : AbstractChunkingStrategy
        Defaults to :class:`SectionChunkingStrategy` (one chunk per section).
    embedder : AbstractChunkEmbedder | None
        When set, ``embed()`` is called on all chunks after chunking.
    """

    enabled: bool = True
    chunking_strategy: AbstractChunkingStrategy = Field(
        default_factory=SectionChunkingStrategy
    )
    embedder: Optional[AbstractChunkEmbedder] = None

    model_config = {"arbitrary_types_allowed": True}


# --------------------------------------------------------------------------- #
# PDFDocumentConfig                                                            #
# --------------------------------------------------------------------------- #

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
        Reliable for title and authors in well-tagged PDFs; often noisy for
        documents originally produced from Word.
    ``DOCLING``
        Infers fields from the Docling structural IR: first section header /
        first plausible line for title; abstract/summary section for summary.
    ``LLM``
        Sends the first ~60 lines of the document to the configured LLM and
        parses a JSON response for authors and acknowledgements.
        Requires ``llm_base_url`` and ``llm_api_key`` to be set.

    Defaults
    --------
    - ``title``   : FILE_METADATA → DOCLING
    - ``authors`` : FILE_METADATA → LLM
    - ``summary`` : DOCLING
    - ``publishing_institute`` : FILE_METADATA
    - ``acknowledgements`` : LLM

    Usage
    -----
    Minimal — only ``folder_path`` is required::

        src = PDFSource()
        src.connect({"folder_path": "/data/papers"})

    With LLM enrichment for authors::

        src.connect({
            "folder_path":  "/data/papers",
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key":  "ollama",
            "llm_model":    "gemma2:9b",
        })

    Disable LLM entirely — FILE_METADATA + DOCLING only::

        src.connect({
            "folder_path": "/data/papers",
            "authors": FieldExtractionConfig(
                strategies=OrderedStrategyConfig(
                    order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.DOCLING]
                )
            ),
            "acknowledgements": FieldExtractionConfig(enabled=False),
        })

    With chunking and embedding::

        from database_builder_libs.utility.chunk.summary_and_sections import SummaryAndSectionsStrategy
        from database_builder_libs.utility.embed_chunk.openai_compatible import OpenAICompatibleChunkEmbedder

        src.connect({
            "folder_path": "/data/papers",
            "sections": SectionsConfig(
                chunking_strategy=SummaryAndSectionsStrategy(),
                embedder=OpenAICompatibleChunkEmbedder(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    model="nomic-embed-text",
                ),
            ),
        })
    """

    folder_path: Path

    # ── metadata field configs ──────────────────────────────────────────────

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
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.DOCLING]
            )
        )
    )

    publishing_institute: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.FILE_METADATA]
            )
        )
    )

    acknowledgements: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.LLM]
            )
        )
    )

    # ── section / chunking / embedding ─────────────────────────────────────

    sections: SectionsConfig = Field(default_factory=SectionsConfig)

    # ── LLM connection ──────────────────────────────────────────────────────

    llm_base_url: Optional[str] = None
    llm_api_key:  Optional[str] = None
    llm_model:    str = "gpt-4.1-mini"


# --------------------------------------------------------------------------- #
# PDFSource                                                                    #
# --------------------------------------------------------------------------- #

_SUMMARY_HEADERS = frozenset({"abstract", "summary", "executive summary", "samenvatting"})
_PDF_META_JUNK   = frozenset({"unknown", "untitled", "microsoft word", "writer", "author"})
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

    Synchronisation semantics
    -------------------------
    - ``get_list_artefacts()`` uses file-system mtime for incremental sync.
    - Deletions are not reported.
    - Identifiers are stable as long as files are not moved.

    Lifecycle
    ---------
    ``connect()`` must be called before any other method.
    """

    _config: Optional[PDFDocumentConfig] = PrivateAttr(default=None)
    _parser: Optional[DocumentParserDocling] = PrivateAttr(default=None)
    _llm_client: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def _connect_impl(self, config: Mapping[str, Any]) -> None:
        """
        Validate config, initialise the Docling parser, and build the LLM
        client if ``llm_base_url`` and ``llm_api_key`` are provided.

        Raises
        ------
        ValueError
            If ``folder_path`` does not exist or is not a directory.
        """
        self._config = PDFDocumentConfig(**config)
        if not self._config.folder_path.is_dir():
            raise ValueError(
                f"folder_path '{self._config.folder_path}' does not exist or is not a directory."
            )
        self._parser = DocumentParserDocling()
        self._llm_client = self._build_llm_client()
        logger.info(f"PDFSource connected to folder: {self._config.folder_path}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_all_documents_metadata(self, limit: int = -1) -> List[dict[str, Any]]:
        """
        Return lightweight file-level metadata for all PDFs in the folder.

        No Docling conversion is performed.  Intended for quick inventory.

        Returns one dict per PDF with keys:
        ``id``, ``path``, ``size``, ``modified``, ``pdf_meta``.
        """
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

    def get_list_artefacts(
        self, last_synced: Optional[datetime]
    ) -> list[tuple[str, datetime]]:
        """
        Return PDFs modified after ``last_synced``, sorted ascending by mtime.

        Parameters
        ----------
        last_synced : datetime | None
            UTC timestamp of last sync.  ``None`` returns all files.
        """
        self._ensure_connected()
        assert self._config is not None

        if last_synced is not None and last_synced.tzinfo is None:
            last_synced = last_synced.replace(tzinfo=timezone.utc)

        artefacts: list[tuple[str, datetime]] = []
        for pdf_path in self._config.folder_path.rglob("*.pdf"):
            modified = datetime.fromtimestamp(pdf_path.stat().st_mtime, tz=timezone.utc)
            if last_synced is None or modified > last_synced:
                artefacts.append((
                    str(pdf_path.relative_to(self._config.folder_path)),
                    modified,
                ))

        artefacts.sort(key=lambda t: t[1])
        logger.info(f"get_list_artefacts: {len(artefacts)} PDF(s) since {last_synced}.")
        return artefacts

    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """
        Run the full pipeline for each artefact and return one Content per file.

        Pipeline
        --------
        1. **Parse** – Docling converts the PDF to a ``ParsedDocument``.
        2. **Metadata** – per-field strategy cascade populates
           :class:`DocumentMetadata`.
        3. **Chunking** – configured ``AbstractChunkingStrategy`` produces
           ``Chunk`` objects from the parsed sections.
        4. **Embedding** – configured ``AbstractChunkEmbedder`` (if any)
           populates ``Chunk.vector``.
        5. **Assembly** – everything is packed into ``Content``.

        Errors in any step are caught and logged; a ``Content`` is always
        returned so one bad file does not abort the batch.

        Raises
        ------
        RuntimeError
            If called before ``connect()``.
        KeyError
            If an artefact no longer exists on disk.
        """
        self._ensure_connected()
        assert self._config is not None
        assert self._parser is not None

        contents: list[Content] = []

        for relative_id, modified in artefacts:
            pdf_path = self._config.folder_path / relative_id
            if not pdf_path.exists():
                raise KeyError(f"Artefact '{relative_id}' no longer exists.")

            stat        = pdf_path.stat()
            pdf_path_str = str(pdf_path.resolve())

            # 1. Parse
            parsed: Optional[ParsedDocument] = None
            num_pages: Optional[int] = None
            try:
                parsed    = self._parser.parse(pdf_path_str)
                num_pages = len(parsed.doc.pages) if parsed.doc.pages else None
            except DocumentConversionError as exc:
                logger.warning(f"Docling conversion failed for '{relative_id}': {exc}")
            except Exception as exc:
                logger.warning(f"Unexpected parse error for '{relative_id}': {exc}")

            # 2. Metadata
            metadata = self._extract_metadata(pdf_path_str=pdf_path_str, parsed=parsed)

            # 3. Chunking
            chunks: list[Chunk] = []
            if self._config.sections.enabled and parsed is not None:
                chunks = self._chunk(parsed=parsed, document_id=relative_id)

            # 4. Embedding
            if chunks and self._config.sections.embedder is not None:
                chunks = self._embed(chunks)

            # 5. Assemble
            contents.append(Content(
                date=modified,
                id_=relative_id,
                content={
                    "file_path":      pdf_path_str,
                    "file_name":      pdf_path.name,
                    "file_size":      stat.st_size,
                    "num_pages":      num_pages,
                    "pdf_meta":       self._read_pdf_meta(pdf_path_str),
                    "metadata":       dataclasses.asdict(metadata),
                    **self._structural_meta(parsed),
                    "chunks":         [dataclasses.asdict(c) for c in chunks],
                },
            ))
            logger.debug(f"Processed '{relative_id}': {len(chunks)} chunk(s).")

        logger.info(f"get_content returning {len(contents)} Content object(s).")
        return contents

    # ------------------------------------------------------------------ #
    # Metadata extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_metadata(
        self,
        *,
        pdf_path_str: str,
        parsed: Optional[ParsedDocument],
    ) -> DocumentMetadata:
        """
        Populate :class:`DocumentMetadata` by running each field's configured
        strategy order, stopping on first success per field.

        Each field consults its ``FieldExtractionConfig.strategies.order`` list
        and calls the matching strategy method:

        - ``FILE_METADATA`` → pypdf embedded metadata
        - ``DOCLING``       → structural heuristics on the Docling IR
        - ``LLM``           → JSON extraction via the configured LLM

        Returns an empty :class:`DocumentMetadata` if *parsed* is ``None``
        (conversion failed) or if all strategies fail.
        """
        assert self._config is not None
        meta = DocumentMetadata()

        if parsed is None:
            return meta

        try:
            # Cache expensive operations computed at most once per document.
            _lines:     Optional[List[str]]  = None
            _pdf_info:  Optional[Any]        = None
            _llm_result: Optional[dict[str, Any]] = None

            def lines() -> List[str]:
                nonlocal _lines
                if _lines is None:
                    _lines = self._first_lines(parsed.doc, limit=120)
                return _lines

            def pdf_info() -> Any:
                nonlocal _pdf_info
                if _pdf_info is None:
                    try:
                        _pdf_info = PdfReader(pdf_path_str).metadata or {}
                    except Exception:
                        _pdf_info = {}
                return _pdf_info

            def llm_result() -> dict[str, Any]:
                nonlocal _llm_result
                if _llm_result is None:
                    _llm_result = self._call_llm(lines()) if self._llm_client else {}
                return _llm_result

            # ── title ─────────────────────────────────────────────────────
            if self._config.title.enabled:
                for strategy in self._config.title.strategies.order:
                    if strategy == ExtractionStrategy.FILE_METADATA:
                        raw = self._clean_meta_string(
                            getattr(pdf_info(), "title", None)
                            or pdf_info().get("/Title")
                        )
                        if raw:
                            meta.title = raw
                            meta.source["title"] = "pdf_metadata"
                    elif strategy == ExtractionStrategy.DOCLING:
                        val = self._first_section_header(parsed.doc) \
                              or self._first_reasonable_line(lines())
                        if val:
                            meta.title = val
                            meta.source["title"] = "docling_heuristic"
                    elif strategy == ExtractionStrategy.LLM and self._llm_client:
                        val = llm_result().get("title")
                        if val:
                            meta.title = val
                            meta.source["title"] = "llm"
                    if meta.title and self._config.title.strategies.stop_on_success:
                        break

            # ── authors ───────────────────────────────────────────────────
            if self._config.authors.enabled:
                for strategy in self._config.authors.strategies.order:
                    if strategy == ExtractionStrategy.FILE_METADATA:
                        raw = self._clean_meta_string(
                            getattr(pdf_info(), "author", None)
                            or pdf_info().get("/Author")
                        )
                        if raw:
                            meta.authors = self._split_authors(raw)
                            meta.source["authors"] = "pdf_metadata"
                    elif strategy == ExtractionStrategy.DOCLING:
                        for line in lines()[:10]:
                            parsed_authors = self._parse_author_line(line)
                            if len(parsed_authors) >= 2:
                                meta.authors = parsed_authors
                                meta.source["authors"] = "docling_heuristic"
                                break
                    elif strategy == ExtractionStrategy.LLM and self._llm_client:
                        authors = llm_result().get("authors")
                        if authors:
                            meta.authors = authors
                            meta.source["authors"] = "llm"
                    if meta.authors and self._config.authors.strategies.stop_on_success:
                        break

            # ── summary ───────────────────────────────────────────────────
            if self._config.summary.enabled:
                for strategy in self._config.summary.strategies.order:
                    if strategy == ExtractionStrategy.DOCLING:
                        val = self._find_summary(parsed.doc)
                        if val:
                            meta.summary = val
                            meta.source["summary"] = "docling_heuristic"
                    elif strategy == ExtractionStrategy.LLM and self._llm_client:
                        val = llm_result().get("summary")
                        if val:
                            meta.summary = val
                            meta.source["summary"] = "llm"
                    if meta.summary and self._config.summary.strategies.stop_on_success:
                        break

            # ── publishing_institute ──────────────────────────────────────
            if self._config.publishing_institute.enabled:
                for strategy in self._config.publishing_institute.strategies.order:
                    if strategy == ExtractionStrategy.FILE_METADATA:
                        raw = self._clean_meta_string(
                            pdf_info().get("Producer") or pdf_info().get("/Producer")
                            or pdf_info().get("Creator") or pdf_info().get("/Creator")
                        )
                        if raw and not any(j in raw.lower() for j in _PDF_META_JUNK):
                            meta.publishing_institute = Institution(name=raw)
                            meta.source["publishing_institute"] = "pdf_metadata"
                    elif strategy == ExtractionStrategy.LLM and self._llm_client:
                        val = llm_result().get("publishing_institute")
                        if val:
                            meta.publishing_institute = Institution(name=val)
                            meta.source["publishing_institute"] = "llm"
                    if meta.publishing_institute and \
                            self._config.publishing_institute.strategies.stop_on_success:
                        break

            # ── acknowledgements ──────────────────────────────────────────
            if self._config.acknowledgements.enabled:
                for strategy in self._config.acknowledgements.strategies.order:
                    if strategy == ExtractionStrategy.LLM and self._llm_client:
                        raw_acks = llm_result().get("acknowledgements", [])
                        acks = [
                            Acknowledgement(
                                name=a.get("name", ""),
                                type=a.get("type", ""),
                                relation=a.get("relation", ""),
                            )
                            for a in raw_acks
                            if a.get("name")
                        ]
                        if acks:
                            meta.acknowledgements = acks
                            meta.source["acknowledgements"] = "llm"
                    if meta.acknowledgements and \
                            self._config.acknowledgements.strategies.stop_on_success:
                        break

        except Exception as exc:
            logger.warning(f"Metadata extraction failed for '{pdf_path_str}': {exc}")

        return meta

    # ------------------------------------------------------------------ #
    # LLM                                                                  #
    # ------------------------------------------------------------------ #

    def _build_llm_client(self) -> Optional[Any]:
        """
        Instantiate an OpenAI-compatible client from the config, or ``None``.

        Returns ``None`` (with a warning) if ``llm_base_url`` / ``llm_api_key``
        are unset or if the ``openai`` package is not installed.
        """
        assert self._config is not None
        if not self._config.llm_base_url or not self._config.llm_api_key:
            return None
        try:
            from openai import OpenAI
            return OpenAI(
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
            )
        except ImportError:
            logger.warning("openai package not installed; LLM extraction skipped.")
            return None

    def _call_llm(self, lines: List[str]) -> dict[str, Any]:
        """
        Send the first 60 lines to the LLM and parse the JSON response.

        Returns a dict with keys ``authors``, ``acknowledgements``, and
        optionally ``summary`` and ``publishing_institute``.  Returns an empty
        dict on any failure.
        """
        assert self._config is not None
        text = "\n".join(lines[:60])
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
{text}
"""
        try:
            res = self._llm_client.chat.completions.create(
                model=self._config.llm_model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = res.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                return {}
            return json.loads(match.group(0))
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")
            return {}

    # ------------------------------------------------------------------ #
    # Docling structural helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _first_lines(doc: Any, limit: int) -> List[str]:
        out: List[str] = []
        for node, _ in doc.iterate_items():
            if isinstance(node, (SectionHeaderItem, TextItem)):
                for ln in (node.text or "").strip().splitlines():
                    ln = ln.strip()
                    if ln:
                        out.append(ln)
                        if len(out) >= limit:
                            return out
        return out

    # Known repository/watermark strings that are not real titles.
    _TITLE_NOISE = frozenset({
        "university of huddersfield repository",
        "original citation",
        "repository",
        "preprint",
        "author's accepted manuscript",
        "accepted manuscript",
        "post-print",
    })

    @classmethod
    def _looks_like_title(cls, t: str) -> bool:
        """Return True if *t* looks like a genuine document title."""
        if len(t) < 8 or "@" in t or len(t.split()) < 2:
            return False
        # Skip strings that are entirely uppercase — usually section labels,
        # not titles (the actual title in all-caps will be caught by LLM).
        if t == t.upper() and len(t) > 30:
            return False
        if t.strip().lower() in cls._TITLE_NOISE:
            return False
        return True

    @classmethod
    def _first_section_header(cls, doc: Any) -> Optional[str]:
        for node, _ in doc.iterate_items():
            if isinstance(node, SectionHeaderItem):
                t = (node.text or "").strip()
                if cls._looks_like_title(t):
                    return t
        return None

    @classmethod
    def _first_reasonable_line(cls, lines: List[str]) -> Optional[str]:
        for ln in lines[:30]:
            if cls._looks_like_title(ln):
                return ln
        return None

    @staticmethod
    def _find_summary(doc: Any) -> Optional[str]:
        collecting = False
        collected: List[str] = []
        for node, _ in doc.iterate_items():
            if isinstance(node, SectionHeaderItem):
                h = (node.text or "").strip().lower()
                if any(k in h for k in _SUMMARY_HEADERS):
                    collecting = True
                    continue
                if collecting:
                    break
            if collecting and isinstance(node, TextItem):
                t = (node.text or "").strip()
                if t:
                    collected.append(t)
        return "\n".join(collected[:4]) or None

    # ------------------------------------------------------------------ #
    # pypdf / author helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_pdf_meta(pdf_path_str: str) -> dict[str, Any]:
        try:
            reader = PdfReader(pdf_path_str)
            if reader.metadata:
                return {k.lstrip("/"): v for k, v in reader.metadata.items() if v is not None}
        except Exception as exc:
            logger.warning(f"Could not read PDF metadata for '{pdf_path_str}': {exc}")
        return {}

    @staticmethod
    def _clean_meta_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in _PDF_META_JUNK:
            return None
        # reject strings that look like tool-generated titles
        if s.lower().startswith("microsoft word"):
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
        text = text.replace(" and ", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        pattern = re.compile(r"^([A-Z][a-zA-Z\-']+)\s+([A-Z])\.?$")
        authors = []
        for part in parts:
            m = pattern.match(part)
            if m:
                last, initial = m.groups()
                authors.append(f"{initial} {last}")
        return authors

    # ------------------------------------------------------------------ #
    # Chunking / embedding                                                 #
    # ------------------------------------------------------------------ #

    def _chunk(self, *, parsed: ParsedDocument, document_id: str) -> list[Chunk]:
        assert self._config is not None
        try:
            summary = self._find_summary(parsed.doc)
            chunks = self._config.sections.chunking_strategy.chunk(
                parsed.sections,
                document_id=document_id,
                summary=summary,
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

    # ------------------------------------------------------------------ #
    # Structural meta                                                      #
    # ------------------------------------------------------------------ #

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