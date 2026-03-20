from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import DoclingDocument
from docling.datamodel.document import ErrorItem

from database_builder_libs.utility.extract.document_parser_docling import (
    ConversionFault,
    DocumentConversionError,
    DocumentParserDocling,
    ParsedDocument,
)


def make_conversion_error() -> DocumentConversionError:
    fault = ConversionFault(
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
    return DocumentConversionError(faults=[fault])


def _stub_converter(parser: DocumentParserDocling, doc: MagicMock) -> None:
    """Wire parser._converter so convert() returns a result wrapping *doc*."""
    fake_result = MagicMock()
    fake_result.document = doc
    fake_result.errors = None
    fake_result.hash = "abc123"
    doc.pages = [MagicMock()]
    doc.iterate_items.return_value = iter([])
    parser._converter.convert.return_value = fake_result


@pytest.fixture
def parser():
    instance = DocumentParserDocling()
    instance._converter = MagicMock()
    return instance


@pytest.fixture
def docling_doc():
    return MagicMock(spec=DoclingDocument)


@pytest.fixture(scope="session")
def pdf_bytes():
    pdf = Path(__file__).parent / "data" / "ProjectProposalTest.pdf"
    return pdf.read_bytes()


def test_parse_stream_returns_parsed_document(parser, docling_doc):
    """A successful conversion returns a ParsedDocument instance."""
    _stub_converter(parser, docling_doc)

    result = parser.parse_stream(name="test.pdf", stream=BytesIO(b"fake pdf"))

    assert isinstance(result, ParsedDocument)


def test_parse_stream_passes_correct_arguments(parser, docling_doc):
    """The converter is called with the correct file size cap, error flag, and stream name."""
    _stub_converter(parser, docling_doc)

    parser.parse_stream(name="invoice.pdf", stream=BytesIO(b"123"))

    call_kwargs = parser._converter.convert.call_args.kwargs
    assert call_kwargs["max_file_size"] == 67_108_864
    assert call_kwargs["raises_on_error"] is False
    assert call_kwargs["source"].name == "invoice.pdf"


def test_parse_stream_carries_correct_name(parser, docling_doc):
    """The name passed to parse_stream is preserved on the returned ParsedDocument."""
    _stub_converter(parser, docling_doc)

    result = parser.parse_stream(name="report.pdf", stream=BytesIO(b"x"))

    assert result.name == "report.pdf"


def test_parse_stream_invalid_extension_raises_value_error(parser):
    """An unsupported file extension raises a ValueError before conversion is attempted."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        parser.parse_stream(name="proposal.xyz", stream=BytesIO(b"x"))


def test_parse_stream_empty_document_raises_conversion_error(parser):
    """A conversion result with no pages raises a DocumentConversionError."""
    fake_result = MagicMock()
    fake_result.document = MagicMock()
    fake_result.document.pages = []
    fake_result.errors = []
    fake_result.hash = "dead"
    parser._converter.convert.return_value = fake_result

    with pytest.raises(DocumentConversionError):
        parser.parse_stream(name="broken.pdf", stream=BytesIO(b"bad"))


def test_parse_stream_conversion_error_faults_populated(parser):
    """Errors reported by the converter are wrapped into faults on the raised DocumentConversionError."""
    fake_result = MagicMock()
    fake_result.document = MagicMock()
    fake_result.document.pages = []
    fake_result.errors = [
        ErrorItem(
            component_type="pipeline",
            module_name="pytest",
            error_message="oops",
        )
    ]
    fake_result.hash = "dead"
    parser._converter.convert.return_value = fake_result

    with pytest.raises(DocumentConversionError) as exc_info:
        parser.parse_stream(name="broken.pdf", stream=BytesIO(b"bad"))

    assert len(exc_info.value.faults) == 1
    assert exc_info.value.faults[0].hashvalue == "dead"


def test_parse_file_not_found_raises(tmp_path):
    """Parsing a path that does not exist raises a FileNotFoundError."""
    p = DocumentParserDocling()
    with pytest.raises(FileNotFoundError):
        p.parse(str(tmp_path / "missing.pdf"))


def test_parse_file_on_disk_returns_parsed_document(parser, docling_doc, tmp_path):
    """A valid file on disk is parsed and its filename is reflected in the result."""
    _stub_converter(parser, docling_doc)

    pdf = tmp_path / "dummy.pdf"
    pdf.write_bytes(b"fake")

    result = parser.parse(str(pdf))

    assert isinstance(result, ParsedDocument)
    assert result.name == "dummy.pdf"


def test_parse_stream_exposes_doc(parser, docling_doc):
    """The underlying DoclingDocument is accessible via result.doc."""
    _stub_converter(parser, docling_doc)

    result = parser.parse_stream(name="test.pdf", stream=BytesIO(b"x"))

    assert result.doc is docling_doc


def test_parse_stream_empty_doc_has_empty_collections(parser, docling_doc):
    """A document with no items produces empty lists for all content collections."""
    _stub_converter(parser, docling_doc)

    result = parser.parse_stream(name="empty.pdf", stream=BytesIO(b"x"))

    assert result.sections == []
    assert result.tables == []
    assert result.figures == []
    assert result.code_blocks == []
    assert result.list_blocks == []
    assert result.footnotes == []
    assert result.furniture == []


def test_conversion_fault_fields_accessible():
    """ConversionFault exposes hashvalue, path_file_document, and faults as expected."""
    error = ErrorItem(
        component_type="pipeline",
        module_name="test",
        error_message="fail",
    )
    fault = ConversionFault(
        faults=[error],
        hashvalue="abc",
        path_file_document=Path("broken.pdf"),
    )
    assert fault.hashvalue == "abc"
    assert fault.path_file_document == Path("broken.pdf")
    assert len(fault.faults) == 1


def test_document_conversion_error_message_contains_path():
    """The string representation of a DocumentConversionError includes the offending file path."""
    fault = ConversionFault(
        faults=[],
        hashvalue="xyz",
        path_file_document=Path("bad.pdf"),
    )
    err = DocumentConversionError(faults=[fault])
    assert "bad.pdf" in str(err)


def test_document_conversion_error_faults_attribute():
    """The faults list passed at construction is preserved on the exception."""
    fault = ConversionFault(faults=[], hashvalue="xyz", path_file_document=Path("bad.pdf"))
    err = DocumentConversionError(faults=[fault])
    assert err.faults == [fault]


def test_document_conversion_error_is_value_error():
    """DocumentConversionError is a subclass of ValueError."""
    err = make_conversion_error()
    assert isinstance(err, ValueError)


def test_vectorize_returns_parsed_document(pdf_bytes):
    """End-to-end: parsing a real PDF returns a ParsedDocument."""
    parser = DocumentParserDocling()
    result = parser.parse_stream(name="proposal.pdf", stream=BytesIO(pdf_bytes))
    assert isinstance(result, ParsedDocument)


def test_vectorize_doc_is_docling_document(pdf_bytes):
    """End-to-end: the doc attribute on the result is a genuine DoclingDocument."""
    parser = DocumentParserDocling()
    result = parser.parse_stream(name="proposal.pdf", stream=BytesIO(pdf_bytes))
    assert isinstance(result.doc, DoclingDocument)


def test_vectorize_extracts_expected_text(pdf_bytes):
    """End-to-end: known headings from the test PDF appear somewhere in the extracted sections."""
    parser = DocumentParserDocling()
    result = parser.parse_stream(name="proposal.pdf", stream=BytesIO(pdf_bytes))

    all_text = " ".join(text for _, text, _ in result.sections)
    assert "Projecttitel" in all_text or "Samenvatting" in all_text or "Werkpakket A" in all_text


def test_vectorize_sections_are_non_empty(pdf_bytes):
    """End-to-end: at least one section is extracted from the test PDF."""
    parser = DocumentParserDocling()
    result = parser.parse_stream(name="proposal.pdf", stream=BytesIO(pdf_bytes))
    assert len(result.sections) > 0


def test_vectorize_sections_have_non_empty_text(pdf_bytes):
    """End-to-end: every extracted section has a string title and non-blank body text."""
    parser = DocumentParserDocling()
    result = parser.parse_stream(name="proposal.pdf", stream=BytesIO(pdf_bytes))
    for title, text, tables in result.sections:
        assert isinstance(title, str)
        assert isinstance(text, str)
        assert text.strip(), f"Section '{title}' has empty text"


def test_vectorize_consumes_input_stream(pdf_bytes):
    """End-to-end: seeking back to the start of a consumed stream allows a second successful parse."""
    parser = DocumentParserDocling()
    stream = BytesIO(pdf_bytes)

    result1 = parser.parse_stream(name="proposal.pdf", stream=stream)
    assert isinstance(result1, ParsedDocument)

    stream.seek(0)
    result2 = parser.parse_stream(name="proposal.pdf", stream=stream)
    assert isinstance(result2, ParsedDocument)


def test_vectorize_invalid_extension_returns_error(pdf_bytes):
    """End-to-end: an unsupported extension raises a ValueError even with real PDF bytes."""
    parser = DocumentParserDocling()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        parser.parse_stream(name="proposal.xyz", stream=BytesIO(pdf_bytes))


def test_vectorize_corrupted_pdf_raises_conversion_error():
    """End-to-end: bytes that are not a valid PDF trigger a DocumentConversionError."""
    parser = DocumentParserDocling()
    with pytest.raises(DocumentConversionError):
        parser.parse_stream(name="broken.pdf", stream=BytesIO(b"%PDF totally broken file"))