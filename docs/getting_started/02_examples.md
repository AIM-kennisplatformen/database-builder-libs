# Examples

This page provides practical examples of how to use the database-builder-libs library for various use cases.

## Table of Contents

- [Working with Zotero Source](#working-with-zotero-source)
- [Document Vectorization](#document-vectorization)
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

## Document Vectorization

The library provides utilities to convert documents into vector representations for semantic search.

### Initializing the Vectorizer

```python
from database_builder_libs.utility.embedding.vectorize_document import VectorizeDocument

# Initialize the document vectorizer
vectorizer = VectorizeDocument()
```

### Vectorizing a Document

```python
import io

# Open a document file
with open("path/to/document.pdf", "rb") as f:
    document_data = io.BytesIO(f.read())

# Vectorize the document
document_name = "document.pdf"
result = vectorizer.vectorize(document_name, document_data)

# Check if vectorization was successful
if hasattr(result, "faultss"):
    print(f"Error vectorizing document: {result}")
else:
    # Process the DoclingDocument
    print(f"Successfully vectorized document with {len(result.pages)} pages")
    
    # Access document content
    for page in result.pages:
        print(f"Page {page.page_num}: {len(page.blocks)} blocks")
        
        # Access text blocks
        for block in page.blocks:
            if hasattr(block, "text"):
                print(f"Text: {block.text}")
```

### Handling Different Document Types

```python
# The vectorizer supports multiple document formats
supported_formats = [
    ".csv", ".docx", ".html", ".md", ".pdf", ".pptx", ".xlsx"
]

print(f"Supported document formats: {', '.join(supported_formats)}")
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