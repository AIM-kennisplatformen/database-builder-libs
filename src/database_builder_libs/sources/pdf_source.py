from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from enum import Enum
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pypdf import PdfReader

from database_builder_libs.models.abstract_source import AbstractSource, Content
from database_builder_libs.models.abstract_chunk_embedder import AbstractChunkEmbedder
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.utility.chunk.n_points_section import SectionChunkingStrategy
from database_builder_libs.utility.extract.document_parser_docling import (
    DocumentConversionError,
    DocumentParserDocling,
    ParsedDocument,
)


class ExtractionStrategy(str, Enum):
    DOCLING = "docling"
    LLM = "llm"
    FILE_METADATA = "file_metadata"


class StrategyConfig(BaseModel):
    timeout: Optional[int] = None
    confidence_threshold: Optional[float] = None


class OrderedStrategyConfig(BaseModel):
    order: List[ExtractionStrategy] = Field(default_factory=list)
    stop_on_success: bool = True
    configs: Dict[ExtractionStrategy, StrategyConfig] = Field(default_factory=dict)

    @field_validator("order")
    def no_duplicates(cls, v: list) -> list:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate strategies in order")
        return v


class FieldExtractionConfig(BaseModel):
    enabled: bool = True
    strategies: OrderedStrategyConfig = Field(default_factory=OrderedStrategyConfig)


class SectionsConfig(BaseModel):
    """
    Controls section extraction, chunking, and optional embedding.

    Attributes
    ----------
    enabled : bool
        When ``False`` no chunks are produced and ``Content.content["chunks"]``
        will be an empty list.
    chunking_strategy : AbstractChunkingStrategy
        Strategy used to convert ``ParsedDocument.sections`` into ``Chunk``
        objects.  Defaults to :class:`SectionChunkingStrategy` (one chunk per
        section).
    embedder : AbstractChunkEmbedder | None
        When provided, its ``embed()`` method is called on the produced chunks
        to populate their ``vector`` fields before assembly.  When ``None``
        chunks are returned with empty vectors.
    """

    enabled: bool = True
    chunking_strategy: AbstractChunkingStrategy = Field(
        default_factory=SectionChunkingStrategy
    )
    embedder: Optional[AbstractChunkEmbedder] = None

    model_config = {"arbitrary_types_allowed": True}


class PDFDocumentConfig(BaseModel):
    """
    Configuration for the PDF source pipeline.
 
    Each ``FieldExtractionConfig`` controls which extraction engines are tried
    for that metadata field, in order, stopping on first success when
    ``stop_on_success`` is ``True``.
 
    Attributes
    ----------
    folder_path : Path
        Root directory scanned recursively for ``*.pdf`` files.
    title, authors, authors_institute, summary, abstract,
    publishing_institute, acknowledgements : FieldExtractionConfig
        Per-field extraction configuration with ordered strategy lists.
    sections : SectionsConfig
        Controls section extraction, chunking strategy, and optional embedding.
 
    Usage
    -----
    **Minimal — defaults only**
 
    The simplest valid config only requires ``folder_path``.  All metadata
    fields use their built-in strategy defaults and sections are extracted
    with one chunk per section, no embedding::
 
        config = PDFDocumentConfig(folder_path=Path("/data/papers"))
 
    **Connecting PDFSource**
 
    Pass the config as a plain dict to ``PDFSource.connect()``::
 
        src = PDFSource()
        src.connect({
            "folder_path": "/data/papers",
        })
 
    **Customising metadata extraction strategies**
 
    Each metadata field accepts an ``enabled`` flag and an ordered list of
    ``ExtractionStrategy`` values.  The pipeline tries each strategy in order
    and stops on the first that produces a result (``stop_on_success=True``).
 
    Available strategies:
 
    - ``ExtractionStrategy.FILE_METADATA`` — reads embedded PDF metadata via
      ``pypdf`` (fast, no ML required).
    - ``ExtractionStrategy.DOCLING`` — uses Docling's structural extraction
      (section headers, abstract detection).
    - ``ExtractionStrategy.LLM`` — sends the first page to an LLM for
      extraction (requires ``OPENAI_API_KEY`` in the environment).
 
    Example — use only ``FILE_METADATA`` for title, disable
    ``authors_institute`` entirely, and try ``DOCLING`` then ``LLM`` for
    summary::
 
        config = PDFDocumentConfig(
            folder_path=Path("/data/papers"),
            title=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(
                    order=[ExtractionStrategy.FILE_METADATA],
                )
            ),
            authors_institute=FieldExtractionConfig(enabled=False),
            summary=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(
                    order=[ExtractionStrategy.DOCLING, ExtractionStrategy.LLM],
                    stop_on_success=True,
                )
            ),
        )
 
    **Customising chunking strategy**
 
    Supply any ``AbstractChunkingStrategy`` implementation via
    ``SectionsConfig.chunking_strategy``.  The default is
    ``SectionChunkingStrategy`` (one chunk per Docling section).
 
    Switch to fixed-size windows of 800 characters::
 
        from database_builder_libs.utility.chunk.n_points_fixed_size import (
            FixedSizeChunkingStrategy,
        )
 
        config = PDFDocumentConfig(
            folder_path=Path("/data/papers"),
            sections=SectionsConfig(
                chunking_strategy=FixedSizeChunkingStrategy(chunk_size=800),
            ),
        )
 
    Use the summary-and-sections strategy (prepends an abstract chunk, then
    splits the rest into 5 evenly-sized body chunks)::
 
        from database_builder_libs.utility.chunk.summary_and_sections import (
            SummaryAndSectionsStrategy,
        )
 
        config = PDFDocumentConfig(
            folder_path=Path("/data/papers"),
            sections=SectionsConfig(
                chunking_strategy=SummaryAndSectionsStrategy(min_chars=30),
            ),
        )
 
    **Adding an embedder**
 
    Attach any ``AbstractChunkEmbedder`` to ``SectionsConfig.embedder``.
    Chunks are embedded in a single batched call immediately after chunking.
 
    OpenAI-compatible endpoint (e.g. Ollama running locally)::
 
        from database_builder_libs.utility.embed_chunk.openai_compatible import (
            OpenAICompatibleChunkEmbedder,
        )
 
        config = PDFDocumentConfig(
            folder_path=Path("/data/papers"),
            sections=SectionsConfig(
                chunking_strategy=SectionChunkingStrategy(),
                embedder=OpenAICompatibleChunkEmbedder(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    model="nomic-embed-text",
                ),
            ),
        )
 
    Local HuggingFace transformer model::
 
        from database_builder_libs.utility.embed_chunk.transformer_based import (
            TransformersChunkEmbedder,
        )
 
        config = PDFDocumentConfig(
            folder_path=Path("/data/papers"),
            sections=SectionsConfig(
                embedder=TransformersChunkEmbedder(
                    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                    batch_size=64,
                ),
            ),
        )
 
    **Disabling section extraction**
 
    Set ``sections.enabled = False`` to skip chunking entirely.  The
    ``Content.content["chunks"]`` list will be empty but all other metadata
    fields are still extracted::
 
        config = PDFDocumentConfig(
            folder_path=Path("/data/papers"),
            sections=SectionsConfig(enabled=False),
        )
    """
 
    folder_path: Path
 
    title: FieldExtractionConfig = Field(default_factory=FieldExtractionConfig)
 
    authors: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.LLM]
            )
        )
    )
 
    authors_institute: FieldExtractionConfig = Field(default_factory=FieldExtractionConfig)
 
    summary: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.DOCLING, ExtractionStrategy.LLM]
            )
        )
    )
 
    abstract: FieldExtractionConfig = Field(
        default_factory=lambda: FieldExtractionConfig(
            strategies=OrderedStrategyConfig(
                order=[ExtractionStrategy.DOCLING, ExtractionStrategy.LLM]
            )
        )
    )
 
    publishing_institute: FieldExtractionConfig = Field(default_factory=FieldExtractionConfig)
    acknowledgements: FieldExtractionConfig = Field(default_factory=FieldExtractionConfig)
    sections: SectionsConfig = Field(default_factory=SectionsConfig)
 
 

class PDFSource(AbstractSource):
    """
    PDF document source implementation of AbstractSource.

    Provides incremental synchronisation of a folder of PDF files and exposes
    each file as a canonical ``Content`` object containing extracted metadata,
    structured sections, and optionally embedded ``Chunk`` objects.

    Mapping
    -------
    PDF file path (relative)  →  Content.id_
    File mtime (UTC)          →  Content.date
    Extracted payload         →  Content.content

    Content.content keys
    --------------------
    ``file_path``   – absolute path on disk
    ``file_name``   – basename
    ``file_size``   – bytes
    ``num_pages``   – page count from pypdf (``None`` on read error)
    ``pdf_meta``    – raw embedded PDF metadata dict from pypdf
    ``sections``    – list of ``(title, text, tables)`` dicts from Docling
    ``chunks``      – list of serialised ``Chunk`` dicts, ready for indexing

    Synchronisation semantics
    -------------------------
    - ``get_list_artefacts()`` performs incremental sync using file-system mtime.
    - Files with ``mtime > last_synced`` are returned; deletions are not reported.
    - Returned timestamps are timezone-aware UTC.
    - Identifiers (relative paths) are stable across runs as long as files are
      not renamed or moved.

    Lifecycle
    ---------
    ``connect()`` must be called before any other method.
    """

    _config: Optional[PDFDocumentConfig] = PrivateAttr(default=None)
    _parser: Optional[DocumentParserDocling] = PrivateAttr(default=None)

    def _connect_impl(self, config: Mapping[str, Any]) -> None:
        """
        Validate config and initialise the Docling parser.

        Parameters
        ----------
        config : Mapping[str, Any]
            Must contain at least ``folder_path`` pointing to an existing
            directory.  All other keys map to ``PDFDocumentConfig`` fields.

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
        logger.info(f"PDFSource connected to folder: {self._config.folder_path}")

    def get_all_documents_metadata(self, limit: int = -1) -> List[dict[str, Any]]:
        """
        Retrieve file-level metadata for all PDFs in the configured folder.

        Recursively scans ``folder_path`` for ``*.pdf`` files and reads their
        embedded PDF metadata via ``pypdf``.  No Docling conversion is
        performed; this method is intended for quick inventory purposes.

        Parameters
        ----------
        limit : int, optional
            Maximum number of documents to return.  ``-1`` (default) means no limit.

        Returns
        -------
        list[dict[str, Any]]
            One dict per PDF with keys: ``id``, ``path``, ``size``,
            ``modified``, ``pdf_meta``.
        """
        self._ensure_connected()
        assert self._config is not None

        results: List[dict[str, Any]] = []

        for pdf_path in sorted(self._config.folder_path.rglob("*.pdf")):
            if limit != -1 and len(results) >= limit:
                break

            stat = pdf_path.stat()
            pdf_meta: dict[str, Any] = {}
            try:
                reader = PdfReader(str(pdf_path))
                if reader.metadata:
                    pdf_meta = {
                        k.lstrip("/"): v
                        for k, v in reader.metadata.items()
                        if v is not None
                    }
            except Exception as exc:
                logger.warning(f"Could not read PDF metadata for '{pdf_path}': {exc}")

            results.append({
                "id":       str(pdf_path.relative_to(self._config.folder_path)),
                "path":     pdf_path,
                "size":     stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                "pdf_meta": pdf_meta,
            })

        return results

    def get_list_artefacts(
        self, last_synced: Optional[datetime]
    ) -> list[tuple[str, datetime]]:
        """
        Return PDF files modified after ``last_synced``.

        Parameters
        ----------
        last_synced : datetime | None
            UTC timestamp of the last successful sync.
            If ``None``, all discovered PDF files are returned.

        Returns
        -------
        list[tuple[str, datetime]]
            ``(relative_path, mtime_utc)`` pairs, sorted ascending by mtime.

        Sync guarantees
        ---------------
        - Identifiers are stable across runs as long as files are not moved.
        - Timestamps are timezone-aware UTC.
        - Newly created and modified files are included; deletions are not reported.
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
        logger.info(
            f"get_list_artefacts: {len(artefacts)} new/modified PDF(s) "
            f"since {last_synced}."
        )
        return artefacts

    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """
        Parse and enrich PDFs, returning one ``Content`` per artefact.

        Pipeline per file
        -----------------
        1. **Parse** – ``DocumentParserDocling.parse()`` converts the PDF to a
           ``ParsedDocument`` containing the Docling IR, sections, tables,
           figures, footnotes, and furniture.
        2. **Metadata** – embedded PDF metadata is read via ``pypdf`` and merged
           with structural metadata from the ``ParsedDocument`` (page count,
           section titles, figure/table counts).
        3. **Chunking** – ``SectionsConfig.chunking_strategy.chunk()`` converts
           ``ParsedDocument.sections`` into ``Chunk`` objects.  The summary
           extracted from Docling's abstract/summary section is forwarded so
           strategies like :class:`SummaryAndSectionsStrategy` can use it.
        4. **Embedding** – when ``SectionsConfig.embedder`` is set, its
           ``embed()`` method populates each ``Chunk.vector`` in one batched
           call.
        5. **Assembly** – everything is packed into a ``Content`` object.

        Errors in steps 1–4 are caught and logged; a ``Content`` is always
        returned so a single bad file does not abort the batch.

        Parameters
        ----------
        artefacts : list[tuple[str, datetime]]
            As returned by :meth:`get_list_artefacts`.

        Returns
        -------
        list[Content]
            One ``Content`` per artefact in the same order.

        Raises
        ------
        RuntimeError
            If called before :meth:`connect`.
        KeyError
            If an artefact's path no longer exists on disk.
        """
        self._ensure_connected()
        assert self._config is not None
        assert self._parser is not None

        contents: list[Content] = []

        for relative_id, modified in artefacts:
            pdf_path = self._config.folder_path / relative_id

            if not pdf_path.exists():
                raise KeyError(
                    f"Artefact '{relative_id}' no longer exists at '{pdf_path}'."
                )

            stat = pdf_path.stat()
            pdf_path_str = str(pdf_path.resolve())

            # ── 1. Parse ─────────────────────────────────────────────────────
            parsed: Optional[ParsedDocument] = None
            num_pages: Optional[int] = None
            try:
                parsed = self._parser.parse(pdf_path_str)
                num_pages = len(parsed.doc.pages) if parsed.doc.pages else None
            except DocumentConversionError as exc:
                logger.warning(f"Docling conversion failed for '{relative_id}': {exc}")
            except Exception as exc:
                logger.warning(f"Unexpected parse error for '{relative_id}': {exc}")

            # ── 2. Metadata ───────────────────────────────────────────────────
            pdf_meta = self._read_pdf_meta(pdf_path_str)
            structural_meta = self._structural_meta(parsed)

            # ── 3. Chunking ───────────────────────────────────────────────────
            chunks: list[Chunk] = []
            if self._config.sections.enabled and parsed is not None:
                chunks = self._chunk(parsed=parsed, document_id=relative_id)

            # ── 4. Embedding ──────────────────────────────────────────────────
            if chunks and self._config.sections.embedder is not None:
                chunks = self._embed(chunks)

            # ── 5. Assemble ───────────────────────────────────────────────────
            contents.append(Content(
                date=modified,
                id_=relative_id,
                content={
                    "file_path":       pdf_path_str,
                    "file_name":       pdf_path.name,
                    "file_size":       stat.st_size,
                    "num_pages":       num_pages,
                    "pdf_meta":        pdf_meta,
                    **structural_meta,
                    "chunks":          [dataclasses.asdict(c) for c in chunks],
                },
            ))
            logger.debug(
                f"Processed '{relative_id}': {len(chunks)} chunk(s)."
            )

        logger.info(f"get_content returning {len(contents)} Content object(s).")
        return contents

    def _read_pdf_meta(self, pdf_path_str: str) -> dict[str, Any]:
        """
        Read embedded PDF metadata via ``pypdf``.

        Returns an empty dict on any read error (logged as a warning) so that
        a corrupt or password-protected PDF does not abort the pipeline.
        """
        try:
            reader = PdfReader(pdf_path_str)
            if reader.metadata:
                return {
                    k.lstrip("/"): v
                    for k, v in reader.metadata.items()
                    if v is not None
                }
        except Exception as exc:
            logger.warning(f"Could not read PDF metadata for '{pdf_path_str}': {exc}")
        return {}

    @staticmethod
    def _structural_meta(parsed: Optional[ParsedDocument]) -> dict[str, Any]:
        """
        Derive lightweight structural metadata from a ``ParsedDocument``.

        Returns counts and titles that are cheap to compute and useful for
        filtering without loading the full chunk list.  Returns all-zero /
        empty values when *parsed* is ``None`` (i.e. conversion failed).
        """
        if parsed is None:
            return {
                "num_sections": 0,
                "num_tables":   0,
                "num_figures":  0,
                "section_titles": [],
            }
        return {
            "num_sections":   len(parsed.sections),
            "num_tables":     len(parsed.tables),
            "num_figures":    len(parsed.figures),
            "section_titles": [title for title, _, _ in parsed.sections if title],
        }

    def _chunk(
        self,
        *,
        parsed: ParsedDocument,
        document_id: str,
    ) -> list[Chunk]:
        """
        Apply the configured ``AbstractChunkingStrategy`` to parsed sections.

        The summary is derived from the first section whose title matches
        common abstract/summary header names and forwarded to the strategy so
        that :class:`SummaryAndSectionsStrategy` (and similar) can prepend it
        as chunk 0.

        Returns an empty list on any error (logged as a warning).
        """
        assert self._config is not None
        try:
            summary = self._find_summary(parsed)
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
        """
        Populate ``Chunk.vector`` for each chunk via the configured embedder.

        Returns the original chunks unchanged (with a warning) if the embedder
        raises, so downstream indexing can still store text-only records.
        """
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
    def _find_summary(parsed: ParsedDocument) -> Optional[str]:
        """
        Return the body text of the first abstract / summary section, or ``None``.

        Matches section titles case-insensitively against a small set of common
        header names used in academic and technical documents.
        """
        _SUMMARY_HEADERS = {"abstract", "summary", "executive summary", "samenvatting"}
        for title, text, _ in parsed.sections:
            if title.strip().lower() in _SUMMARY_HEADERS:
                return text.strip() or None
        return None