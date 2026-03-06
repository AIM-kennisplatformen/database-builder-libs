# Embedding (Vectorization)

## Overview

The VectorizeDocument utility provides document-to-text conversion capabilities using the Docling library. It supports multiple document formats (PDF, Word, Excel, HTML, Markdown, etc.) and prepares content for downstream embedding and chunking operations.

## Design Notes
### Processing Flow

The VectorizeDocument follows this processing pattern:

1. **Format Validation**:
    - Check file extension against allowed formats
    - Reject unsupported formats early

2. **Document Conversion**:
    - Create document stream from input bytes
    - Apply format-specific conversion options
    - Handle timeouts and size limits

3. **Error Handling**:
    - Capture conversion faults
    - Report empty documents as failures
    - Provide detailed error context

### Configuration Details

* **PDF Processing**: Uses PyPdfium backend with optional OCR (disabled by default)
* **Timeout**: 180 seconds per document to handle large files
* **OCR Languages**: English and Dutch when OCR is enabled
* **File Size Limit**: 64MB maximum to prevent memory issues

## Docstring vectorize document

::: database_builder_libs.utility.embedding.vectorize_document.VectorizeDocument
    handler: python