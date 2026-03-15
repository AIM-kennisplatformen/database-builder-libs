from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pprint import pformat
from typing import IO, Sequence
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocumentStream, ErrorItem
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    PipelineOptions,
)
from docling.document_converter import (
    CsvFormatOption,
    DocumentConverter,
    ExcelFormatOption,
    HTMLFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)

from docling_core.types.doc import DoclingDocument
from pathlib import Path

_ALLOWED_FORMATS = [
    InputFormat.CSV,
    InputFormat.DOCX,
    InputFormat.HTML,
    InputFormat.MD,
    InputFormat.PDF,
    InputFormat.PPTX,
    InputFormat.XLSX,
]

_ALLOWED_EXTENSIONS = {f".{fmt.value}" for fmt in _ALLOWED_FORMATS}


@dataclass
class Faultss:
    faults: Sequence[ErrorItem]
    hashvalue: str
    path_file_document: PathLike[str]


class PipelineDocumentsConversionFailedError(ValueError):
    def __init__(self, *, faultss: Sequence[Faultss]) -> None:
        self.faultss = faultss
        messages = []
        for faults in self.faultss:
            messages.append(
                f"Failed to convert raw document '{faults.path_file_document!s}' ({faults.hashvalue}) "
                f"because of {len(faults.faults)} fault(s): {pformat(faults.faults)}"
            )
        super().__init__("\n".join(messages))


class VectorizeDocument:
    def __init__(
        self,
        *,
        path_dir_artifacts: str | None = None,
    ) -> None:
        """Initialize the raw document to Docling document conversion pipeline.

        Configures a document converter with support for multiple file formats and processing settings. Sets up CSV,
        PDF, Microsoft Word, HTML, Markdown, Microsoft PowerPoint, and Microsoft Excel document conversion.

        Args:
            path_dir_artifacts: Optional path from which to source predictive models.
        """
        pdfpipelineoptions = PdfPipelineOptions(
            artifacts_path=path_dir_artifacts,
            do_ocr=False,
            document_timeout=180,
            ocr_options=EasyOcrOptions(download_enabled=False, lang=["en", "nl"]),
        )
        pipelineoptions = PipelineOptions(document_timeout=180)
        self.documentconverter = DocumentConverter(
            allowed_formats=_ALLOWED_FORMATS,
            format_options={
                InputFormat.CSV: CsvFormatOption(pipeline_options=pipelineoptions),
                InputFormat.DOCX: WordFormatOption(pipeline_options=pipelineoptions),
                InputFormat.HTML: HTMLFormatOption(pipeline_options=pipelineoptions),
                InputFormat.MD: MarkdownFormatOption(pipeline_options=pipelineoptions),
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdfpipelineoptions,
                    backend=PyPdfiumDocumentBackend,
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_options=pipelineoptions
                ),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=pipelineoptions),
            },
        )

    def vectorize(
        self, name_document: str, data_document: IO[bytes]
    ) -> DoclingDocument | PipelineDocumentsConversionFailedError:
        suffix = Path(name_document).suffix.lower()

        if suffix not in _ALLOWED_EXTENSIONS:
            fault = Faultss(
                faults=[],
                hashvalue="unsupported-extension",
                path_file_document=Path(name_document),
            )
            return PipelineDocumentsConversionFailedError(faultss=[fault])

        conversionresult = self.documentconverter.convert(
            max_file_size=67_108_864,
            source=DocumentStream(
                name=name_document, stream=BytesIO(data_document.read())
            ),
            raises_on_error=False,
        )
        return next(
            self._process_conversionresults(
                conversionresults=(conversionresult,),
            )
        )

    def _process_conversionresults(self, *, conversionresults):
        for result in conversionresults:
            # Docling always returns a document object
            doc = getattr(result, "document", None)
            errors = getattr(result, "errors", None)

            # Failure conditions
            is_empty = not doc or (hasattr(doc, "pages") and len(doc.pages) == 0)

            if errors or is_empty:
                fault = Faultss(
                    faults=errors or [],
                    hashvalue=getattr(result, "hash", "unknown"),
                    path_file_document=Path(getattr(doc, "name", "unknown")),
                )
                yield PipelineDocumentsConversionFailedError(faultss=[fault])
            else:
                yield doc
