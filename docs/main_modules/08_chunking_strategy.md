# Chunking strategies

## Overview

The **AbstractChunkingStrategy** class defines the contract that all concrete chunking
implementations must satisfy. It encapsulates the transformation of raw document sections
into a flat, ordered list of `Chunk` objects that are ready for embedding and indexing.
By adhering to this interface, different splitting approaches — section-based, fixed-size,
sliding-window, or summary-prefixed — can be swapped interchangeably while the rest of
the pipeline remains agnostic to the underlying strategy.

---

## Design notes

### Interaction pattern

`AbstractChunkingStrategy` follows a single-phase transformation pattern:

1. **Input** — an ordered sequence of `RawSection` tuples, each carrying a section title,
   body text, and any tables extracted from that section. These are produced directly by
   `DocumentParserDocling` and passed in without further transformation.
2. **Chunking** — the strategy splits, merges, or partitions the sections according to its
   own logic and returns a flat list of `Chunk` objects.
3. **Output** — every returned `Chunk` has a non-empty `text`, a stable `chunk_index`
   starting from 0, the caller-supplied `document_id`, and an empty `vector` list.
   Embedding is a separate downstream concern.

### Choosing a strategy

| Class | Chunks produced | When to use |
|---|---|---|
| `SectionChunkingStrategy` | One per section | Clean heading structure; sections are already semantically coherent |
| `FixedSizeChunkingStrategy` | One or more per section | Uniform context window needed; no overlap required |
| `SlidingWindowChunkingStrategy` | More than fixed-size due to overlap | Boundary recall matters; dense technical text with cross-boundary sentences |
| `SummaryAndSectionsStrategy` | One per section (+ 1 summary if provided) | Section structure must be preserved with an optional LLM-generated summary chunk prepended |

---

## Common chunk fields

Every `Chunk` returned by any strategy has the following fields:

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | The `document_id` passed to `.chunk()`, unchanged. |
| `chunk_index` | `int` | Monotonically increasing from 0. |
| `text` | `str` | Non-empty chunk body. |
| `vector` | `list` | Always `[]` until the embedding stage populates it. |
| `metadata` | `dict` | Strategy-specific; see each strategy below. |

---

## Implementations

### SectionChunkingStrategy

Produces exactly one chunk per non-empty document section. This is the simplest strategy
and maps cleanly onto the heading structure that Docling extracts. It is the right default
when sections are already semantically coherent units such as academic papers or reports.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_chars` | `int` | `20` | Sections shorter than this after stripping are silently dropped, preventing index pollution from stub sections such as lone headings with no body. |
| `include_title_in_text` | `bool` | `False` | When `True`, the section title is prepended to the chunk text as `"<title>\n<text>"`. Useful when the title adds retrieval signal that does not appear in the body. |

**Metadata fields**

| Key | Type | Description |
|---|---|---|
| `section_title` | `str` | Title of the source section. |
| `has_tables` | `bool` | `True` if the section contained at least one table. |

---

### FixedSizeChunkingStrategy

Splits each section's text into non-overlapping fixed-size character windows. Each section
may produce one or more chunks depending on its length relative to `chunk_size`. Splits are
made on whitespace boundaries wherever possible to avoid cutting words mid-token.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `1000` | Target maximum number of characters per chunk. |
| `min_chars` | `int` | `20` | Windows shorter than this are dropped; typically catches the last fragment of a short section. |

**Metadata fields**

| Key | Type | Description |
|---|---|---|
| `section_title` | `str` | Title of the source section. |
| `has_tables` | `bool` | `True` if the source section contained at least one table. |

---

### SlidingWindowChunkingStrategy

Produces overlapping character windows across each section's text. Overlapping windows
preserve cross-boundary context that non-overlapping splits lose, at the cost of a larger
index and some retrieval redundancy. Useful for dense technical text where important
sentences often span what would otherwise be a hard split boundary.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `1000` | Target maximum number of characters per window. |
| `overlap` | `int` | `200` | Number of characters shared between consecutive windows. Must be strictly less than `chunk_size`. |
| `min_chars` | `int` | `20` | Windows shorter than this are dropped. |

**Raises**

| Exception | Condition |
|---|---|
| `ValueError` | `overlap >= chunk_size`. |

**Metadata fields**

| Key | Type | Description |
|---|---|---|
| `section_title` | `str` | Title of the source section. |
| `has_tables` | `bool` | `True` if the source section contained at least one table. |

---

### SummaryAndSectionsStrategy

Produces one optional summary chunk followed by one chunk per non-empty section, preserving
the document's natural heading structure. Unlike `SummaryAndNSectionsStrategy`, section
boundaries and titles are never merged or discarded — each section maps to exactly one body
chunk. Use this strategy when section-level retrieval granularity must be preserved and an
optional LLM-generated summary chunk is desired at index 0.

**Chunk layout**

```
index 0      →  summary text            (only when summary is provided and non-blank)
index 1..N   →  one chunk per section   (in document order)
```

```
index 0..N   →  one chunk per section   (when no summary is provided)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_chars` | `int` | `20` | Sections shorter than this after stripping are silently dropped. |

**Metadata fields — summary chunk**

| Key | Type | Description |
|---|---|---|
| `chunk_type` | `str` | Always `"summary"`. |

**Metadata fields — body chunks**

| Key | Type | Description |
|---|---|---|
| `chunk_type` | `str` | Always `"body"`. |
| `section_title` | `str` | Title of the source section. |
| `has_tables` | `bool` | `True` if the source section contained at least one table. |

---

## Docstrings AbstractChunkStrategy

::: database_builder_libs.models.abstract_chunk_strategy
    handler: python

## Docstrings SectionChunkingStrategy

::: database_builder_libs.utility.chunk.n_points_section
    handler: python

## Docstrings FixedSizeChunkingStrategy

::: database_builder_libs.utility.chunk.n_points_fixed_size
    handler: python

## Docstrings SlidingWindowChunkingStrategy

::: database_builder_libs.utility.chunk.n_points_sliding_window
    handler: python

## Docstrings SummaryAndSectionsStrategy

::: database_builder_libs.utility.chunk.summary_and_sections
    handler: python