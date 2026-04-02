# PDF source

## Overview

The PDF source allows you to parse, extract metadata from, chunk, and embed PDF documents stored in a local folder. It implements the AbstractSource interface to provide incremental synchronization based on file modification times, and exposes a fully configurable extraction pipeline that can combine embedded PDF metadata, structural heuristics from Docling, and LLM-based extraction for each metadata field independently.

## Design Notes

### Interaction Patterns

The PDFSource follows these interaction patterns:

1. **Connection Pattern**:
    - Validate folder path exists on disk
    - Initialise the Docling document parser
    - Build the LLM client if credentials are provided
    - All field extraction strategies are configured at connect time

2. **Synchronization Pattern**:
    - Scan folder recursively for `.pdf` files
    - Compare file mtime against `last_synced` cursor
    - Return stable relative paths as artefact identifiers
    - Sorted ascending by mtime for deterministic ordering

3. **Content Retrieval Pattern**:
    - Parse each PDF with Docling to produce a structural IR
    - Run the metadata extraction pipeline (see Configuration below)
    - Chunk the parsed sections using the configured strategy
    - Embed chunks if an embedder is configured
    - Pack everything into a `Content` object

4. **Metadata Extraction Pattern**:
    - Each field (`title`, `authors`, `summary`, `publishing_institute`, `acknowledgements`) has its own ordered strategy list
    - Strategies are tried in order; extraction stops on first success when `stop_on_success=True`
    - Expensive operations (LLM call, PDF metadata read) are cached and run at most once per document

### Implementation Details

* **Incremental Sync**: Uses filesystem mtime for change detection. Deletions are not reported.
* **Stable Identifiers**: Artefact IDs are paths relative to `folder_path` and remain stable as long as files are not moved.
* **Parser Resilience**: Docling conversion failures are caught and logged per file; the pipeline continues and returns a `Content` with empty chunks and metadata for that file.
* **LLM Caching**: The LLM is called at most once per document regardless of how many fields request it. The result is cached and reused across fields.
* **PDF Metadata Robustness**: The embedded PDF metadata reader handles both plain dicts and `DocumentInformation` objects (pypdf), as well as gracefully dropping any non-string keys caused by library version mismatches.
* **Chunking**: Sections produced by Docling are passed to the configured `AbstractChunkingStrategy`. Chunking failures are caught and logged without aborting the pipeline.
* **Embedding**: If an `AbstractChunkEmbedder` is configured, it is called on the chunk list after chunking. Embedding failures return the original un-embedded chunks.

## Configuring the PDFSource

Configuration is passed as a dict to `src.connect({...})` and validated into a `PDFDocumentConfig` internally. All keys except `folder_path` are optional.

### Minimal setup

Only `folder_path` is required. With no other config, metadata extraction uses embedded PDF metadata and Docling heuristics only — no LLM calls are made.

```python
src = PDFSource()
src.connect({"folder_path": "/data/papers"})
```

### Enabling LLM extraction

Provide `llm_base_url`, `llm_api_key`, and optionally `llm_model` to enable LLM-based extraction. The LLM receives the first 60 lines of the document and is expected to return JSON with `title`, `authors`, `publishing_institute`, and `acknowledgements`.

```python
src = PDFSource()
src.connect({
    "folder_path":  "/data/papers",
    "llm_base_url": "http://localhost:11434/v1",
    "llm_api_key":  "ollama",
    "llm_model":    "gemma2:9b",
})
```

Any OpenAI-compatible endpoint is supported. The default model is `gpt-4.1-mini`.

### Configuring extraction strategies per field

Each metadata field can be configured independently using a `FieldExtractionConfig` that holds an `OrderedStrategyConfig`. The three available strategies are:

| Strategy | Key | Description |
|---|---|---|
| `ExtractionStrategy.FILE_METADATA` | `"file_metadata"` | Reads embedded PDF metadata via pypdf. Fast, no ML. Reliable for well-tagged PDFs; often noisy for Word-derived documents. |
| `ExtractionStrategy.DOCLING` | `"docling"` | Infers values from the Docling structural IR. Uses section headers for title, abstract section detection for summary. |
| `ExtractionStrategy.LLM` | `"llm"` | Sends the document header to the configured LLM. Most flexible but requires LLM credentials and adds latency. |

The `OrderedStrategyConfig` controls the order strategies are tried and whether to stop on the first success:

```python
from database_builder_libs.sources.pdf_source import (
    FieldExtractionConfig,
    OrderedStrategyConfig,
    ExtractionStrategy,
)

# Try LLM first, fall back to Docling heuristics
title_config = FieldExtractionConfig(
    strategies=OrderedStrategyConfig(
        order=[ExtractionStrategy.LLM, ExtractionStrategy.DOCLING],
        stop_on_success=True,   # default — stops as soon as one strategy succeeds
    )
)

# Run both strategies regardless of success — LLM overwrites FILE_METADATA result
authors_config = FieldExtractionConfig(
    strategies=OrderedStrategyConfig(
        order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.LLM],
        stop_on_success=False,  # LLM always runs and overwrites
    )
)

# Disable a field entirely
acks_config = FieldExtractionConfig(enabled=False)
```

Pass these configs by field name in the connect dict:

```python
src.connect({
    "folder_path":          "/data/papers",
    "llm_base_url":         "http://localhost:11434/v1",
    "llm_api_key":          "ollama",
    "title":                title_config,
    "authors":              authors_config,
    "acknowledgements":     acks_config,
})
```

### Default extraction strategies

When no field config is provided, PDFSource uses these defaults:

| Field | Default strategy order | Notes |
|---|---|---|
| `title` | `FILE_METADATA` → `DOCLING` | LLM not in default chain; add it if embedded metadata is unreliable |
| `authors` | `FILE_METADATA` → `LLM` | LLM used as fallback when embedded metadata is empty or noisy |
| `summary` | `DOCLING` | Abstract section detection is reliable; LLM rarely needed |
| `publishing_institute` | `FILE_METADATA` | LLM can be added for documents without well-tagged metadata |
| `acknowledgements` | `LLM` | Only LLM can extract these reliably from free text |

### Configuring chunking and embedding

Section chunking and vector embedding are controlled via `SectionsConfig`. Disable chunking entirely, or plug in any `AbstractChunkingStrategy` and `AbstractChunkEmbedder` implementation.

```python
from database_builder_libs.sources.pdf_source import SectionsConfig
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
```

To skip chunking and embedding entirely (metadata and file stats only):

```python
src.connect({
    "folder_path": "/data/papers",
    "sections": SectionsConfig(enabled=False),
})
```

### Skipping fields already known from an external source

When metadata for a document is already available from an external source (e.g. Zotero), pass `FieldExtractionConfig(enabled=False)` explicitly for those fields. Omitting a field from the config is not sufficient — PDFDocumentConfig has defaults for every field and will fall back to them silently.

```python
src.connect({
    "folder_path": "/data/papers",
    "title":   FieldExtractionConfig(enabled=False),  # already known from Zotero
    "authors": FieldExtractionConfig(enabled=False),  # already known from Zotero
    # summary, publishing_institute and acknowledgements will use their defaults
})
```

## Content object structure

`PDFSource.get_content()` returns one `Content` object per artefact. `Content.content` is a dict with the following keys:

| Key | Type | Description |
|---|---|---|
| `file_path` | `str` | Absolute path to the PDF file |
| `file_name` | `str` | Filename only |
| `file_size` | `int` | File size in bytes |
| `num_pages` | `int \| None` | Page count from Docling; `None` if conversion failed |
| `pdf_meta` | `dict` | Raw embedded PDF metadata from pypdf, keys stripped of leading `/` |
| `metadata` | `dict` | `DocumentMetadata` serialised via `dataclasses.asdict()` |
| `num_sections` | `int` | Section count from Docling |
| `num_tables` | `int` | Table count from Docling |
| `num_figures` | `int` | Figure count from Docling |
| `section_titles` | `list[str]` | Ordered list of section header strings |
| `chunks` | `list[dict]` | `Chunk` objects serialised via `dataclasses.asdict()` |

The `metadata` dict mirrors `DocumentMetadata` and includes a `source` sub-dict that maps each populated field to the strategy that filled it, for example:

```json
{
  "title": "A Randomised Controlled Trial ...",
  "authors": ["Heyman, Bob", "Harrington, Barbara"],
  "publishing_institute": {"name": "Housing Studies", "parent": null},
  "summary": "This paper discusses ...",
  "acknowledgements": [
    {"name": "NEARG", "type": "organization", "relation": "collaboration"}
  ],
  "source": {
    "title": "zotero",
    "authors": "zotero",
    "summary": "docling_heuristic",
    "publishing_institute": "llm",
    "acknowledgements": "llm"
  },
  "keywords": null,
  "literature_type": null,
  "strategic_overview": null,
  "target_groups": null,
  "best_practices": null
}
```

## Docstring

::: database_builder_libs.sources.pdf_source
    handler: python