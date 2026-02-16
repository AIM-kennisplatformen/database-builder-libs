from pathlib import Path
import pytest
from io import BytesIO
from unittest.mock import MagicMock
from docling_core.types.doc import DoclingDocument
from backend.utility.embedding.vectorize_document import PipelineDocumentsConversionFailedError, VectorizeDocument

from backend.utility.embedding.vectorize_document import Faultss
from docling.datamodel.document import ErrorItem

def make_failure():
    fault = Faultss(
        faults=[
            ErrorItem(
                component_type="pipeline",
                module_name="pytest",
                error_message="bad file",
            )
        ],
        hashvalue="123",
        path_file_document=Path("broken.pdf"),
    )
    return PipelineDocumentsConversionFailedError(faultss=[fault])

@pytest.fixture
def vectorizer():
    # Create instance normally
    instance = VectorizeDocument()

    # Mock heavy dependencies
    instance.documentconverter = MagicMock()
    instance._process_conversionresults = MagicMock()

    return instance


def test_vectorize_success(vectorizer):
    fake_doc = MagicMock(name="DoclingDocument")
    fake_conversion_result = MagicMock()

    vectorizer.documentconverter.convert.return_value = fake_conversion_result
    vectorizer._process_conversionresults.return_value = iter([fake_doc])

    result = vectorizer.vectorize(
        name_document="test.pdf",
        data_document=BytesIO(b"fake pdf content"),
    )

    # Verify convert was called
    vectorizer.documentconverter.convert.assert_called_once()

    # Verify first yielded result is returned
    assert result == fake_doc

def test_vectorize_passes_correct_arguments(vectorizer):
    vectorizer.documentconverter.convert.return_value = MagicMock()
    vectorizer._process_conversionresults.return_value = iter([MagicMock()])

    vectorizer.vectorize(
        name_document="invoice.pdf",
        data_document=BytesIO(b"123"),
    )

    call_args = vectorizer.documentconverter.convert.call_args
    kwargs = call_args.kwargs

    assert kwargs["max_file_size"] == 67_108_864
    assert kwargs["raises_on_error"] is False
    assert kwargs["source"].name == "invoice.pdf"

@pytest.fixture(scope="session")
def pdf_bytes():
    pdf = Path(__file__).parent / "data" / "ProjectProposalTest.pdf"
    return pdf.read_bytes()

def test_vectorize_returns_docling_document(pdf_bytes):
    vectorizer = VectorizeDocument()

    result = vectorizer.vectorize(
        "proposal.pdf",
        BytesIO(pdf_bytes),
    )

    assert isinstance(result, DoclingDocument)

def test_vectorize_extracts_expected_text(pdf_bytes):
    vectorizer = VectorizeDocument()

    doc = vectorizer.vectorize("proposal.pdf", BytesIO(pdf_bytes))

    text = doc.export_to_markdown()

    assert "Projecttitel" in text
    assert "Samenvatting" in text
    assert "Werkpakket A" in text

def test_vectorize_consumes_input_stream(pdf_bytes):
    vectorizer = VectorizeDocument()

    stream = BytesIO(pdf_bytes)

    doc1 = vectorizer.vectorize("proposal.pdf", stream)
    assert isinstance(doc1, DoclingDocument)

    # Stream now empty → should still succeed because wrapper copies bytes
    stream.seek(0)
    doc2 = vectorizer.vectorize("proposal.pdf", stream)

    assert isinstance(doc2, DoclingDocument)

def test_vectorize_invalid_extension_returns_error(pdf_bytes):
    vectorizer = VectorizeDocument()

    result = vectorizer.vectorize(
        "proposal.xyz",
        BytesIO(pdf_bytes),
    )

    assert isinstance(result, PipelineDocumentsConversionFailedError)

def test_vectorize_corrupted_pdf_returns_error():
    vectorizer = VectorizeDocument()

    broken = BytesIO(b"%PDF totally broken file")

    result = vectorizer.vectorize("broken.pdf", broken)

    assert isinstance(result, PipelineDocumentsConversionFailedError)

def test_vectorize_returns_failure(vectorizer):
    fake_failure = make_failure()

    vectorizer.documentconverter.convert.return_value = MagicMock()
    vectorizer._process_conversionresults.return_value = iter([fake_failure])

    result = vectorizer.vectorize("broken.pdf", BytesIO(b"bad content"))

    assert isinstance(result, PipelineDocumentsConversionFailedError)
