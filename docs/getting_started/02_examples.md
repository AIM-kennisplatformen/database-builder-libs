# Examples

This page provides practical examples of how to use the database-builder-libs library for various use cases.

## Table of Contents

- [Working with Zotero Source](#working-with-zotero-source)
- [Document Parsing](#document-parsing)
- [Chunking Strategies](#chunking-strategies)
- [Using Vector Stores (Qdrant)](#using-vector-stores-qdrant)
- [Using TypeDB Store](#using-typedb-store)

## Working with Zotero Source

The Zotero source allows you to connect to a Zotero library and retrieve documents and metadata.

### Connecting to Zotero

```python
from database_builder_libs.sources.zotero_source import ZoteroSource

# Initialize the Zotero source
zotero_source = ZoteroSource()

# Connect to Zotero with your credentials
zotero_source.connect({
    "library_id": "your_library_id",
    "library_type": "user",  # or "group"
    "api_key": "your_api_key",
    "collection": "optional_collection_id"  # Optional: specific collection
})
```

### Retrieving Modified Items Since Last Sync

```python
from datetime import datetime, timezone

# Get items modified since a specific date (or None for all items)
last_sync = datetime(2023, 1, 1, tzinfo=timezone.utc)
modified_items = zotero_source.get_list_artefacts(last_sync)

print(f"Found {len(modified_items)} modified items")
for item_id, modified_date in modified_items:
    print(f"Item {item_id} was modified at {modified_date}")
```

### Retrieving Content for Items

```python
# Get the full content for the modified items
items_content = zotero_source.get_content(modified_items)

for content in items_content:
    print(f"Item ID: {content.id_}")
    print(f"Modified: {content.date}")
    print(f"Title: {content.content.get('title', 'No title')}")
    print("---")
```

### Downloading Attachments

```python
import os

# Create a directory for downloads
download_dir = "zotero_downloads"
os.makedirs(download_dir, exist_ok=True)

# Download the first attachment for a specific item
item_id = "ABC123"  # Replace with an actual item ID
zotero_source.download_zotero_item(
    item_id=item_id,
    download_path=download_dir
)

# The file will be saved as: zotero_downloads/ABC123.pdf
```

### Getting All Documents from a Collection

```python
# Get all documents from a specific collection
collection_id = "COLLECTION123"  # Replace with an actual collection ID
documents = zotero_source.get_all_documents_metadata(collection_id)

for doc in documents:
    print(f"Document: {doc.get('data', {}).get('title', 'No title')}")
    print(f"Key: {doc.get('data', {}).get('key', 'No key')}")
    print("---")
```

## Document Parsing

`DocumentParserDocling` converts a raw document file into a structured `ParsedDocument`
containing the full Docling IR and all extracted content types (sections, tables, figures,
code blocks, list blocks, footnotes, and page furniture).

### Parsing a file on disk

```python
from database_builder_libs.utility.extract.document_parser_docling import (
    DocumentConversionError,
    DocumentParserDocling,
)

# Instantiate once and reuse for multiple documents.
parser = DocumentParserDocling()

try:
    result = parser.parse("path/to/document.pdf")
except FileNotFoundError:
    print("File not found")
except ValueError as exc:
    print(f"Unsupported format: {exc}")
except DocumentConversionError as exc:
    for fault in exc.faults:
        print(f"Conversion failed: {fault.path_file_document} — {fault.faults}")
```

### Parsing an in-memory stream

Use `parse_stream` when the document is not stored on disk, for example when it has been
downloaded from an API or read from object storage.

```python
from io import BytesIO
from database_builder_libs.utility.extract.document_parser_docling import (
    DocumentConversionError,
    DocumentParserDocling,
)

parser = DocumentParserDocling()

with open("path/to/document.pdf", "rb") as f:
    stream = BytesIO(f.read())

try:
    result = parser.parse_stream(name="document.pdf", stream=stream)
except ValueError as exc:
    print(f"Unsupported format: {exc}")
except DocumentConversionError as exc:
    for fault in exc.faults:
        print(f"Conversion failed: {fault.path_file_document} — {fault.faults}")
```

### Working with the result

`parse` and `parse_stream` both return a `ParsedDocument` — a frozen dataclass whose
fields map directly to content types extracted from the document.

```python
# Sections are the primary input to chunking strategies.
# Each section is a (title, body_text, tables) tuple.
for title, text, tables in result.sections:
    print(f"Section: '{title}'")
    print(f"  {len(text)} characters, {len(tables)} table(s)")

# Tables come with their caption (empty string when absent).
for table in result.tables:
    print(f"Table caption: '{table.caption}'")
    print(table.dataframe)

# Figures come with their caption (empty string when absent).
for figure in result.figures:
    print(f"Figure caption: '{figure.caption}'")

# Code blocks are attributed to their enclosing section.
for block in result.code_blocks:
    print(f"[{block.section_title}] {block.text}")

# List blocks group consecutive list items within the same section.
for block in result.list_blocks:
    print(f"[{block.section_title}]")
    for item in block.items:
        print(f"  - {item}")

# Footnotes in document order.
for footnote in result.footnotes:
    print(f"Footnote: {footnote.text}")

# Page headers and footers, deduplicated across pages.
for entry in result.furniture:
    print(f"{entry.kind}: {entry.text}")

# The full Docling IR is available for any downstream processing
# that needs the raw node graph, bounding boxes, or provenance.
print(type(result.doc))  # <class 'docling_core.types.doc.DoclingDocument'>
```

### Supported formats

```python
supported_formats = [".csv", ".docx", ".html", ".md", ".pdf", ".pptx", ".xlsx"]
```

### Loading ML model artefacts from a local directory

By default Docling downloads its ML models at first use. Pass `path_dir_artifacts` to
load them from a local directory instead, which is useful in air-gapped environments.

```python
parser = DocumentParserDocling(path_dir_artifacts="/opt/docling/models")
result = parser.parse("path/to/document.pdf")
```

## Chunking Strategies

All chunking strategies share the same interface: they accept a list of `RawSection`
tuples — the `sections` field of a `ParsedDocument` — and return a list of `Chunk`
objects. An optional `summary` keyword argument is accepted by every strategy; only
`SummaryAndNSectionsStrategy` makes use of it.

```python
# RawSection = (section_title: str, body_text: str, tables: list[DataFrame])
sections = result.sections  # from DocumentParserDocling.parse / parse_stream
```

### Section chunking

Produces one chunk per section. Sections whose text falls below `min_chars` (default: 20)
are silently dropped. This is the simplest strategy and works well when the source
document has clean, well-scoped headings.

```python
from database_builder_libs.utility.chunk.n_points_section import SectionChunkingStrategy

strategy = SectionChunkingStrategy(
    min_chars=20,               # sections shorter than this are dropped
    include_title_in_text=False # set True to prepend the section title to chunk text
)

chunks = strategy.chunk(sections, document_id="doc-001")

for chunk in chunks:
    print(chunk.chunk_index, chunk.metadata["section_title"], len(chunk.text))
```

With `include_title_in_text=True` the section heading is prepended to the body text,
which can improve retrieval quality for embedding models that benefit from contextual
headers:

```python
strategy = SectionChunkingStrategy(include_title_in_text=True)
chunks = strategy.chunk(sections, document_id="doc-001")
# chunk.text starts with "<section title>\n<body text>"
```

### Fixed-size chunking

Splits each section into non-overlapping windows of at most `chunk_size` characters,
respecting whitespace boundaries. Useful when sections vary wildly in length and a
uniform context window is preferred.

```python
from database_builder_libs.utility.chunk.n_points_fixed_size import FixedSizeChunkingStrategy

strategy = FixedSizeChunkingStrategy(
    chunk_size=500,  # maximum characters per chunk
    min_chars=20,    # windows shorter than this after splitting are dropped
)

chunks = strategy.chunk(sections, document_id="doc-001")

for chunk in chunks:
    print(chunk.chunk_index, chunk.metadata["section_title"], len(chunk.text))
```

Each chunk's `metadata["section_title"]` always reflects the section it was split from,
so provenance is preserved even when a section is split across many chunks.

### Sliding-window chunking

Like fixed-size chunking but consecutive chunks overlap by `overlap` characters, so
context at chunk boundaries is not lost. `overlap` must be strictly less than
`chunk_size`; passing an equal or larger value raises `ValueError`.

```python
from database_builder_libs.utility.chunk.n_points_sliding_window import SlidingWindowChunkingStrategy

strategy = SlidingWindowChunkingStrategy(
    chunk_size=500,  # maximum characters per chunk
    overlap=100,     # characters shared between consecutive chunks; must be < chunk_size
    min_chars=20,
)

chunks = strategy.chunk(sections, document_id="doc-001")
```

Because adjacent chunks share content, this strategy produces more chunks than
fixed-size for the same input. The overlap means the end of chunk *N* and the start
of chunk *N+1* share words, which improves recall for queries that land near a boundary.

### Summary + sections chunking
Preserves the document's natural section structure and optionally prepends a dedicated
summary chunk at index 0. This is the right choice when section-level retrieval
granularity must be maintained and an optional LLM-generated summary is desired.

```python
from database_builder_libs.utility.chunk.summary_and_sections import SummaryAndSectionsStrategy

strategy = SummaryAndSectionsStrategy(min_chars=20)

# Without a summary — produces one chunk per non-empty section.
chunks = strategy.chunk(sections, document_id="doc-001")

for chunk in chunks:
    print(chunk.chunk_index, chunk.metadata["chunk_type"], chunk.metadata["section_title"])
```

Pass a summary string (e.g. produced by an LLM) to prepend a summary chunk. The
summary chunk is assigned chunk_index=0 and all body chunks follow in order. A
whitespace-only summary is treated as absent.

```python
summary_text = "This document covers the annual research results for 2024."

chunks = strategy.chunk(sections, document_id="doc-001", summary=summary_text)

summary_chunk = chunks[0]  # chunk_type == "summary"
body_chunks   = chunks[1:] # chunk_type == "body", one per section

print(summary_chunk.text)
print(f"{len(body_chunks)} body chunks")

# Each body chunk carries its originating section title.
for chunk in body_chunks:
    print(chunk.metadata["section_title"], chunk.metadata["has_tables"])
```

### Choosing a strategy

| Strategy | Chunks produced | When to use |
|---|---|---|
| `SectionChunkingStrategy` | One per section | Clean heading structure, variable section length is acceptable |
| `FixedSizeChunkingStrategy` | One or more per section | Uniform context window needed, no overlap required |
| `SlidingWindowChunkingStrategy` | More than fixed-size due to overlap | Boundary recall matters; willing to trade storage for coverage |
| `SummaryAndSectionsStrategy` | One per section (+ 1 if summary provided) | Section structure must be preserved with an optional summary chunk prepended|

### Common chunk fields

Every `Chunk` returned by any strategy has the following fields:

```python
chunk.document_id   # str  — the document_id passed to .chunk()
chunk.chunk_index   # int  — monotonically increasing from 0
chunk.text          # str  — non-empty chunk body
chunk.vector        # list — always [] until the embedding stage populates it
chunk.metadata      # dict — strategy-specific; always contains "section_title"
```

## Using Vector Stores (Qdrant)

The Qdrant vector store allows you to store and retrieve document chunks based on semantic similarity.

### Connecting to Qdrant

```python
from database_builder_libs.stores.qdrant.qdrant_store import QdrantDatastore
from database_builder_libs.models.chunk import Chunk

# Initialize the Qdrant store
qdrant_store = QdrantDatastore()

# Connect to Qdrant
qdrant_store.connect({
    "url": "http://localhost:6333",  # Qdrant server URL
    "collection": "documents",       # Collection name
    "vector_size": 768               # Embedding dimension size
})
```

### Storing Document Chunks

```python
# Create document chunks with embeddings
chunks = [
    Chunk(
        document_id="doc1",
        chunk_index=0,
        text="This is the first chunk of document 1.",
        vector=[0.1, 0.2, ...],  # Your embedding vector (must match vector_size)
        metadata={"page": 1, "section": "introduction"}
    ),
    Chunk(
        document_id="doc1",
        chunk_index=1,
        text="This is the second chunk of document 1.",
        vector=[0.2, 0.3, ...],
        metadata={"page": 1, "section": "introduction"}
    ),
    # Add more chunks as needed
]

# Store the chunks
qdrant_store.store_chunks(chunks)
```

### Performing Similarity Search

```python
# Create a query vector (must have the same dimension as stored vectors)
query_vector = [0.1, 0.2, ...]  # Your query embedding

# Search for similar chunks
results = qdrant_store.similarity_search(
    vector=query_vector,
    limit=5  # Return top 5 results
)

# Process the results
for chunk in results:
    print(f"Document: {chunk.document_id}")
    print(f"Chunk: {chunk.chunk_index}")
    print(f"Text: {chunk.text}")
    if chunk.metadata:
        print(f"Metadata: {chunk.metadata}")
    print("---")
```

### Retrieving All Chunks for a Document

```python
# Get all chunks for a specific document
document_id = "doc1"
chunks = qdrant_store.get_document_chunks(document_id)

print(f"Found {len(chunks)} chunks for document {document_id}")
```

### Deleting a Document

```python
# Delete all chunks for a document
document_id = "doc1"
deleted_count = qdrant_store.delete_document(document_id)

print(f"Deleted {deleted_count} chunks for document {document_id}")
```

## Using TypeDB Store

The TypeDB store provides a graph database backend for storing and retrieving structured knowledge.

### Connecting to TypeDB

```python
from database_builder_libs.stores.typedb_v2.typedb_v2_store import TypeDbDatastore
from database_builder_libs.models.node import Node, NodeId, EntityType, KeyAttribute

# Initialize the TypeDB store
typedb_store = TypeDbDatastore()

# Connect to TypeDB with schema
typedb_store.connect({
    "uri": "localhost:1729",       # TypeDB server address
    "database": "knowledge_base",  # Database name
    "schema_path": "schema.tql"    # Optional path to schema file
})
```

### Creating and Storing Nodes

```python
# Create a Node representing a person
person_node = Node(
    id=NodeId("person:john_doe"),
    entity_type=EntityType("person"),
    key_attribute=KeyAttribute("email"),
    payload_data={
        "email": "john.doe@example.com",
        "name": "John Doe",
        "age": 30
    },
    relations=[
        {
            "type": "works_for",
            "target": NodeId("organization:acme_corp")
        },
        {
            "type": "authored",
            "target": NodeId("document:report_2023")
        }
    ]
)

# Store the node
typedb_store.store_node(person_node)

# Create and store an organization node
org_node = Node(
    id=NodeId("organization:acme_corp"),
    entity_type=EntityType("organization"),
    key_attribute=KeyAttribute("name"),
    payload_data={
        "name": "Acme Corporation",
        "industry": "Technology",
        "founded": 1990
    }
)

typedb_store.store_node(org_node)

# Create and store a document node
doc_node = Node(
    id=NodeId("document:report_2023"),
    entity_type=EntityType("document"),
    key_attribute=KeyAttribute("title"),
    payload_data={
        "title": "Annual Report 2023",
        "format": "pdf",
        "pages": 42
    }
)

typedb_store.store_node(doc_node)
```

### Retrieving Nodes with Filters

```python
# Retrieve a specific person by email
filter_query = "entity=person&email=john.doe@example.com"
person_nodes = typedb_store.get_nodes(filter_query)

if person_nodes:
    person = person_nodes[0]
    print(f"Found person: {person.payload_data.get('name')}")
    print(f"Email: {person.payload_data.get('email')}")
    print(f"Age: {person.payload_data.get('age')}")
    
    # Print relations
    for relation in person.relations:
        print(f"Relation: {relation.get('type')} -> {relation.get('target')}")

# Retrieve all documents
doc_filter = "entity=document"
documents = typedb_store.get_nodes(doc_filter)

print(f"Found {len(documents)} documents")
for doc in documents:
    print(f"Document: {doc.payload_data.get('title')}")
    print(f"Format: {doc.payload_data.get('format')}")
    print(f"Pages: {doc.payload_data.get('pages')}")
    print("---")

# Retrieve all nodes (canonical representation)
all_nodes = typedb_store.get_nodes(None)
print(f"Total nodes in database: {len(all_nodes)}")
```

### Removing Nodes

```python
# Remove a specific node
try:
    removed_node = typedb_store.remove_node("entity=person&email=john.doe@example.com")
    print(f"Removed node: {removed_node.id}")
except KeyError:
    print("Node not found")
except ValueError as e:
    print(f"Error: {e}")  # Multiple nodes matched the filter
```

These examples demonstrate the core functionality of the database-builder-libs library. You can adapt them to suit your specific use cases.