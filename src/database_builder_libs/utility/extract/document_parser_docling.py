from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from pprint import pformat
from typing import IO, List, NamedTuple, Sequence, Tuple

from pandas import DataFrame
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocumentStream, ErrorItem
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, PipelineOptions
from docling.document_converter import (
    CsvFormatOption, DocumentConverter, ExcelFormatOption, HTMLFormatOption,
    MarkdownFormatOption, PdfFormatOption, PowerpointFormatOption, WordFormatOption,
)
from docling_core.types.doc import (
    CodeItem, ContentLayer, DocItemLabel, DoclingDocument,
    PictureItem, SectionHeaderItem, TableItem, TextItem,
)


_ALLOWED_FORMATS = [
    InputFormat.CSV, InputFormat.DOCX, InputFormat.HTML, InputFormat.MD,
    InputFormat.PDF, InputFormat.PPTX, InputFormat.XLSX,
]
_ALLOWED_EXTENSIONS = {f".{fmt.value}" for fmt in _ALLOWED_FORMATS}


@dataclass(slots=True)
class ConversionFault:
    """Captures pipeline errors for a single document conversion attempt."""

    faults: Sequence[ErrorItem]
    hashvalue: str
    path_file_document: Path


class DocumentConversionError(ValueError):
    """
    Raised when the Docling pipeline fails or produces an empty document.

    Attributes
    ----------
    faults : Sequence[ConversionFault]
        One entry per failed document, containing the raw ``ErrorItem`` list,
        the document hash, and the file path.
    """

    def __init__(self, *, faults: Sequence[ConversionFault]) -> None:
        self.faults = faults
        super().__init__("\n".join(
            f"Failed to convert '{f.path_file_document!s}' ({f.hashvalue}): "
            f"{len(f.faults)} fault(s): {pformat(f.faults)}"
            for f in faults
        ))

# (section_title, body_text, tables_in_section)
RawSection = Tuple[str, str, List[DataFrame]]

class ExtractedTable(NamedTuple):
    """A table extracted from the document body, paired with its caption."""
    caption: str      # empty string when no caption is present
    dataframe: DataFrame

class ExtractedFigure(NamedTuple):
    """A picture/figure extracted from the document body, paired with its caption."""
    caption: str      # empty string when no caption is present

class ExtractedCodeBlock(NamedTuple):
    """A ``CODE``-labelled block, attributed to its enclosing section."""
    text: str
    section_title: str

class ExtractedListBlock(NamedTuple):
    """
    A run of consecutive list items belonging to the same logical list.

    Docling emits individual ``LIST_ITEM`` nodes; consecutive items within the
    same section are grouped here so callers receive complete lists rather than
    isolated bullets.
    """
    items: Tuple[str, ...]
    section_title: str

class ExtractedFootnote(NamedTuple):
    """A ``FOOTNOTE``-labelled text item."""
    text: str

class ExtractedFurniture(NamedTuple):
    """
    A page-level header or footer.

    Detected via ``DocItemLabel.PAGE_HEADER`` / ``PAGE_FOOTER`` (Docling ≥ 1.8)
    or ``ContentLayer.FURNITURE`` (older versions).  Repeated identical strings
    across pages are deduplicated.
    """
    text: str
    kind: str  # "header" | "footer"


@dataclass(frozen=True)
class ParsedDocument:
    """
    Immutable result of converting and structuring a single document.

    Produced by :class:`DocumentParserDocling` and consumed by the rest of the
    pipeline (chunking strategies, metadata extractor, embedder).

    Mapping
    -------
    DoclingDocument node graph  → sections, tables, figures, code_blocks,
                                  list_blocks, footnotes, furniture

    Attributes
    ----------
    doc : DoclingDocument
        Full Docling IR.  Retain for any downstream processing that needs
        access to the raw node graph, bounding boxes, or provenance.
    name : str
        Original filename passed to the converter (e.g. ``"report.pdf"``).
    sections : list[RawSection]
        Body text grouped by section header as ``(title, text, tables)`` tuples.
        The leading nameless section (content before the first header) is
        included when non-empty, with an empty string as its title.
        This is the primary input to chunking strategies.
    tables : list[ExtractedTable]
        All body tables with their captions, in document order.
    figures : list[ExtractedFigure]
        All pictures with their captions, in document order.
    code_blocks : list[ExtractedCodeBlock]
        All ``CODE``-labelled items attributed to their enclosing section.
    list_blocks : list[ExtractedListBlock]
        Consecutive ``LIST_ITEM`` runs grouped per section.
    footnotes : list[ExtractedFootnote]
        All ``FOOTNOTE``-labelled text items, in document order.
    furniture : list[ExtractedFurniture]
        Page headers and footers, deduplicated across pages.
    """
    doc: DoclingDocument
    name: str
    sections: List[RawSection]
    tables: List[ExtractedTable]
    figures: List[ExtractedFigure]
    code_blocks: List[ExtractedCodeBlock]
    list_blocks: List[ExtractedListBlock]
    footnotes: List[ExtractedFootnote]
    furniture: List[ExtractedFurniture]

class DocumentParserDocling:
    """
    Docling implementation of the document parsing pipeline.

    Converts a raw document file into a :class:`ParsedDocument` containing
    the full Docling IR and all structured content extracted in a single pass.

    Mapping
    -------
    Raw file / byte stream  → DoclingDocument  → ParsedDocument

    Extraction pass
    ---------------
    One ``iterate_items(BODY)`` walk produces sections, tables, figures,
    code blocks, list blocks, and footnotes.  A second
    ``iterate_items(BODY + FURNITURE)`` walk collects page headers/footers.

    Supported formats
    -----------------
    PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV.

    Lifecycle
    ---------
    Instantiate once and call :meth:`parse` or :meth:`parse_stream` per document.

    Parameters
    ----------
    path_dir_artifacts : str | None
        When provided, Docling loads ML model artefacts from this directory
        instead of downloading them at runtime.
    """

    def __init__(self, *, path_dir_artifacts: str | None = None) -> None:
        pdf_opts = PdfPipelineOptions(
            artifacts_path=path_dir_artifacts,
            do_ocr=False,
            document_timeout=180,
            ocr_options=EasyOcrOptions(download_enabled=False, lang=["en", "nl"]),
        )
        default_opts = PipelineOptions(document_timeout=180)
        self._converter = DocumentConverter(
            allowed_formats=_ALLOWED_FORMATS,
            format_options={
                InputFormat.CSV:  CsvFormatOption(pipeline_options=default_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=default_opts),
                InputFormat.HTML: HTMLFormatOption(pipeline_options=default_opts),
                InputFormat.MD:   MarkdownFormatOption(pipeline_options=default_opts),
                InputFormat.PDF:  PdfFormatOption(pipeline_options=pdf_opts, backend=PyPdfiumDocumentBackend),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=default_opts),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=default_opts),
            },
        )

    def parse(self, path: str) -> ParsedDocument:
        """
        Convert a file on disk and extract all content types.

        Parameters
        ----------
        path : str
            Absolute or relative path to the document file.

        Returns
        -------
        ParsedDocument

        Raises
        ------
        FileNotFoundError
            If *path* does not point to an existing file.
        ValueError
            If the file extension is not supported.
        DocumentConversionError
            If the Docling pipeline reports errors or produces an empty document.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: '{path}'")
        with file_path.open("rb") as fh:
            return self._convert_and_extract(name=file_path.name, stream=fh)

    def parse_stream(self, name: str, stream: IO[bytes]) -> ParsedDocument:
        """
        Convert an in-memory byte stream and extract all content types.

        Useful when the document is not stored on disk (e.g. downloaded from
        an API or read from object storage).

        Parameters
        ----------
        name : str
            Filename including extension (e.g. ``"report.pdf"``).
            Used by Docling to determine the input format.
        stream : IO[bytes]
            Readable byte stream of the document content.

        Returns
        -------
        ParsedDocument

        Raises
        ------
        ValueError
            If the file extension is not supported.
        DocumentConversionError
            If conversion fails or produces an empty document.
        """
        return self._convert_and_extract(name=name, stream=stream)

    def _convert_and_extract(self, *, name: str, stream: IO[bytes]) -> ParsedDocument:
        suffix = Path(name).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}")

        result = self._converter.convert(
            source=DocumentStream(name=name, stream=BytesIO(stream.read())),
            max_file_size=67_108_864,
            raises_on_error=False,
        )

        doc = getattr(result, "document", None)
        errors = getattr(result, "errors", None)
        is_empty = not doc or (hasattr(doc, "pages") and len(doc.pages) == 0)

        if errors or is_empty:
            raise DocumentConversionError(faults=[ConversionFault(
                faults=errors or [],
                hashvalue=getattr(result, "hash", "unknown"),
                path_file_document=Path(name),
            )])

        assert isinstance(doc, DoclingDocument)
        return self._extract(doc=doc, name=name)

    @staticmethod
    def _extract(*, doc: DoclingDocument, name: str) -> ParsedDocument:
        current_title = ""
        text_buffer: List[str] = []
        section_tables: List[DataFrame] = []
        sections: List[RawSection] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        code_blocks: List[ExtractedCodeBlock] = []
        footnotes: List[ExtractedFootnote] = []
        list_blocks: List[ExtractedListBlock] = []
        list_buffer: List[str] = []

        def flush_section() -> None:
            nonlocal text_buffer, section_tables
            body = "\n".join(text_buffer).strip()
            if body:
                sections.append((current_title, body, section_tables))
            text_buffer, section_tables = [], []

        def flush_list() -> None:
            nonlocal list_buffer
            if list_buffer:
                list_blocks.append(ExtractedListBlock(tuple(list_buffer), current_title))
            list_buffer = []

        for node, _ in doc.iterate_items(included_content_layers={ContentLayer.BODY}):
            label = getattr(node, "label", None)

            if isinstance(node, SectionHeaderItem):
                flush_list()
                flush_section()
                current_title = (node.text or "").strip()

            elif isinstance(node, TableItem):
                flush_list()
                df = node.export_to_dataframe(doc=doc)
                caption = node.caption_text(doc=doc) if hasattr(node, "caption_text") else ""
                tables.append(ExtractedTable(caption or "", df))
                section_tables.append(df)

            elif isinstance(node, PictureItem):
                flush_list()
                caption = node.caption_text(doc=doc) if hasattr(node, "caption_text") else ""
                figures.append(ExtractedFigure(caption or ""))

            elif isinstance(node, CodeItem):
                flush_list()
                if text := (node.text or "").strip():
                    code_blocks.append(ExtractedCodeBlock(text, current_title))

            elif isinstance(node, TextItem):
                text = (node.text or "").strip()
                if label == DocItemLabel.FOOTNOTE:
                    flush_list()
                    if text: 
                        footnotes.append(ExtractedFootnote(text))
                elif label == DocItemLabel.LIST_ITEM:
                    if text: 
                        list_buffer.append(text)
                elif label == DocItemLabel.CODE:
                    flush_list()
                    if text: 
                        code_blocks.append(ExtractedCodeBlock(text, current_title))
                else:
                    flush_list()
                    if text: 
                        text_buffer.append(text)

        flush_list()
        flush_section()

        return ParsedDocument(
            doc=doc, name=name, sections=sections, tables=tables,
            figures=figures, code_blocks=code_blocks, list_blocks=list_blocks,
            footnotes=footnotes, furniture=DocumentParserDocling._extract_furniture(doc),
        )

    @staticmethod
    def _extract_furniture(doc: DoclingDocument) -> List[ExtractedFurniture]:
        furniture: List[ExtractedFurniture] = []
        seen: set[str] = set()

        for node, _ in doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            label = getattr(node, "label", None)
            text = (getattr(node, "text", None) or "").strip()
            if not text:
                continue

            if label == DocItemLabel.PAGE_HEADER:
                kind = "header"
            elif label == DocItemLabel.PAGE_FOOTER:
                kind = "footer"
            elif getattr(node, "content_layer", None) == ContentLayer.FURNITURE:
                kind = "header"  # unclassified furniture defaults to header
            else:
                continue

            key = f"{kind}:{text}"
            if key not in seen:
                seen.add(key)
                furniture.append(ExtractedFurniture(text, kind))

        return furniture