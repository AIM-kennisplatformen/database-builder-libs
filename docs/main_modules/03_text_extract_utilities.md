# Text structure extraction (DocumentParserDocling)

## Overview

`DocumentParserDocling` converts raw document files into structured text using the
[Docling](https://github.com/DS4SD/docling) library. It supports PDF, Word, PowerPoint,
Excel, HTML, Markdown, and CSV, and produces a `ParsedDocument` that is consumed by the
chunking and embedding stages of the pipeline.

---

## Supported formats

| Extension | Format |
|---|---|
| `.pdf` | PDF (PyPdfium backend) |
| `.docx` | Word |
| `.pptx` | PowerPoint |
| `.xlsx` | Excel |
| `.html` | HTML |
| `.md` | Markdown |
| `.csv` | CSV |


## Design notes

### Processing flow

`DocumentParserDocling` follows this processing pattern:

1. **Format validation**
    - The file extension is checked against the allow-list before any I/O takes place.
    - Unsupported extensions raise `ValueError` immediately.

2. **Document conversion**
    - The input (file path or byte stream) is wrapped in a `DocumentStream` and handed to
        Docling's `DocumentConverter`.
    - Format-specific options are applied (PDF uses the PyPdfium backend; all others use the
        default pipeline).
    - A hard file-size cap of 64 MB and a 180-second timeout are enforced on every call.

3. **Error handling**
    - If Docling reports `errors` **or** the resulting document has no pages, a
        `DocumentConversionError` is raised.
    - Each failure is wrapped in a `ConversionFault` that captures the raw `ErrorItem` list,
        the document hash, and the file path, so callers have full context for logging or retry
        logic.

4. **Content extraction**
     
    - A single `iterate_items(BODY)` walk over the `DoclingDocument` node graph populates
            **sections**, **tables**, **figures**, **code blocks**, **list blocks**, and
            **footnotes** in document order.
    - Section text is accumulated in a buffer and flushed whenever a `SectionHeaderItem` or
        the end of the document is reached, producing `(title, body_text, tables)` tuples.
    - Consecutive `LIST_ITEM` nodes within the same section are grouped into a single
        `ExtractedListBlock` rather than emitted as isolated bullets.
    - A second `iterate_items(BODY + FURNITURE)` walk collects page headers and footers,
        deduplicated across pages, into `ExtractedFurniture` entries.

---

## Configuration details

| Setting | Value | Notes |
|---|---|---|
| PDF backend | `PyPdfiumDocumentBackend` | Fast, no native deps |
| OCR | Disabled by default | Enable by passing `do_ocr=True` |
| OCR languages | English, Dutch | Configurable via `EasyOcrOptions` |
| Document timeout | 180 s | Applies to all formats |
| File size limit | 64 MB (`67_108_864` bytes) | Enforced in `convert()` call |

---


## Return type — `ParsedDocument`

`ParsedDocument` is a frozen dataclass. All fields are populated in a single extraction
pass and are immutable after construction.

| Field | Type | Description |
|---|---|---|
| `doc` | `DoclingDocument` | Full Docling IR. Retain for downstream access to the raw node graph, bounding boxes, or provenance. |
| `name` | `str` | Original filename passed to the converter. |
| `sections` | `list[RawSection]` | Body text as `(title, text, tables)` tuples, grouped by section header. The leading nameless section (content before the first header) is included when non-empty, with `""` as its title. Primary input to chunking strategies. |
| `tables` | `list[ExtractedTable]` | All body tables with captions, in document order. |
| `figures` | `list[ExtractedFigure]` | All pictures with captions, in document order. |
| `code_blocks` | `list[ExtractedCodeBlock]` | All `CODE`-labelled items, attributed to their enclosing section. |
| `list_blocks` | `list[ExtractedListBlock]` | Consecutive `LIST_ITEM` runs grouped per section. |
| `footnotes` | `list[ExtractedFootnote]` | All `FOOTNOTE`-labelled items, in document order. |
| `furniture` | `list[ExtractedFurniture]` | Page headers and footers, deduplicated across pages. |

---

## Error types

### `DocumentConversionError(ValueError)`

Raised when the Docling pipeline fails or the output document is empty.

```python
try:
    result = parser.parse_stream(name="report.pdf", stream=stream)
except DocumentConversionError as exc:
    for fault in exc.faults:
        print(fault.path_file_document, fault.hashvalue, fault.faults)
```

### `ConversionFault`

Dataclass attached to `DocumentConversionError.faults`. One entry per failed document.

| Field | Type | Description |
|---|---|---|
| `faults` | `Sequence[ErrorItem]` | Raw Docling error items from the pipeline. |
| `hashvalue` | `str` | Document hash assigned by Docling. |
| `path_file_document` | `Path` | Path or name of the file that failed. |

---

## Docstring vectorize document

::: database_builder_libs.utility.extract.document_parser_docling.DocumentParserDocling
    handler: python