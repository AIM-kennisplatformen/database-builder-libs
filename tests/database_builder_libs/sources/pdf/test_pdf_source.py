import os
import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

from pydantic import Field

from database_builder_libs.models.abstract_chunk_embedder import AbstractChunkEmbedder
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.sources.pdf_source import (
    PDFSource,
    PDFDocumentConfig,
    SectionsConfig,
    ExtractionStrategy,
    FieldExtractionConfig,
    OrderedStrategyConfig,
)
from database_builder_libs.utility.chunk.n_points_section import SectionChunkingStrategy


# --------------------------------------------------------------------------- #
# Test helpers                                                                 #
# --------------------------------------------------------------------------- #

class SpyStrategy(AbstractChunkingStrategy):
    """
    Minimal AbstractChunkingStrategy implementation that records every call
    to chunk() so tests can assert on the arguments it received.

    Used in place of MagicMock where Pydantic's isinstance validation would
    reject a mock object.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def chunk(
        self,
        sections: Sequence[RawSection],
        *,
        document_id: str,
        summary: str | None = None,
    ) -> list[Chunk]:
        self.calls.append({"sections": sections, "document_id": document_id, "summary": summary})
        return []


class SpyEmbedder(AbstractChunkEmbedder):
    """
    Minimal AbstractChunkEmbedder implementation that records calls to embed()
    and returns chunks with a fixed non-empty vector.

    Used in place of MagicMock where Pydantic's model validation would reject
    a mock object as the embedder field on SectionsConfig.

    ``received`` is declared as a Pydantic field (not assigned dynamically) so
    that Pydantic's model validation does not raise on attribute assignment.
    """

    received: list[Chunk] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        self.received = list(chunks)
        return [
            Chunk(
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                vector=[0.1, 0.2, 0.3],
                metadata=c.metadata,
            )
            for c in chunks
        ]


class PDFSourceTests(unittest.TestCase):
    """Tests for PDFSource"""

    cwd = os.path.dirname(os.path.realpath(__file__))

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def connect_source(self, folder: str | Path, **extra) -> PDFSource:
        """Connect a PDFSource to *folder* with the parser mocked out."""
        src = PDFSource()
        src.connect({"folder_path": str(folder), **extra})
        # Replace the real Docling parser with a mock so tests stay fast
        # and do not require ML models to be installed.
        src._parser = MagicMock()
        return src

    def make_parsed_document(
        self,
        sections: list | None = None,
        num_pages: int = 3,
    ) -> MagicMock:
        """Return a ParsedDocument-shaped mock with sensible defaults."""
        parsed = MagicMock()
        parsed.sections = sections or [
            ("Introduction", "This is the introduction text of the document.", []),
            ("Methods", "The methods section describes the approach taken.", []),
            ("Results", "The results section presents the findings.", []),
        ]
        parsed.tables = []
        parsed.figures = []
        parsed.footnotes = []
        parsed.furniture = []
        parsed.doc.pages = [MagicMock()] * num_pages
        return parsed

    def write_dummy_pdf(self, folder: Path, name: str = "paper.pdf") -> Path:
        """Write a zero-byte placeholder so rglob finds a PDF file."""
        path = folder / name
        path.write_bytes(b"")
        return path

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    # ---------------------------------------------------------------------- #
    # connect / _connect_impl                                                 #
    # ---------------------------------------------------------------------- #

    def test_connect_succeeds_with_valid_folder(self):
        """connect() must not raise when folder_path points to an existing directory."""
        src = PDFSource()
        src.connect({"folder_path": str(self.folder)})
        # No exception raised — connection was successful.

    def test_connect_raises_on_missing_folder(self):
        """connect() must raise ValueError when folder_path does not exist."""
        src = PDFSource()
        with self.assertRaises(ValueError):
            src.connect({"folder_path": "/this/path/does/not/exist"})

    def test_connect_is_idempotent(self):
        """Calling connect() twice must not raise or re-initialise the parser."""
        src = PDFSource()
        src.connect({"folder_path": str(self.folder)})
        parser_after_first = src._parser
        src.connect({"folder_path": str(self.folder)})
        self.assertIs(src._parser, parser_after_first)

    def test_methods_raise_before_connect(self):
        """All public methods must raise RuntimeError when called before connect()."""
        src = PDFSource()
        with self.assertRaises(RuntimeError):
            src.get_list_artefacts(None)
        with self.assertRaises(RuntimeError):
            src.get_content([])
        with self.assertRaises(RuntimeError):
            src.get_all_documents_metadata()

    # ---------------------------------------------------------------------- #
    # get_all_documents_metadata                                              #
    # ---------------------------------------------------------------------- #

    def test_get_all_documents_metadata_returns_one_entry_per_pdf(self):
        """get_all_documents_metadata() must return exactly one dict per PDF found.

        The method scans the folder recursively and reads pypdf metadata without
        performing any Docling conversion.  This test verifies the scan logic and
        the structure of the returned dicts.
        """
        self.write_dummy_pdf(self.folder, "paper_a.pdf")
        self.write_dummy_pdf(self.folder, "paper_b.pdf")

        src = self.connect_source(self.folder)
        results = src.get_all_documents_metadata()

        self.assertEqual(len(results), 2)

    def test_get_all_documents_metadata_dict_keys(self):
        """Each metadata dict must contain the mandatory keys."""
        self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        result = src.get_all_documents_metadata()[0]

        for key in ("id", "path", "size", "modified", "pdf_meta"):
            self.assertIn(key, result)

    def test_get_all_documents_metadata_id_is_relative_path(self):
        """The ``id`` key must be the path relative to folder_path, not absolute."""
        sub = self.folder / "subdir"
        sub.mkdir()
        self.write_dummy_pdf(sub, "nested.pdf")

        src = self.connect_source(self.folder)
        result = src.get_all_documents_metadata()[0]

        self.assertEqual(result["id"], "subdir/nested.pdf")
        self.assertFalse(result["id"].startswith("/"))

    def test_get_all_documents_metadata_limit(self):
        """The ``limit`` parameter must cap the number of returned entries."""
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            self.write_dummy_pdf(self.folder, name)

        src = self.connect_source(self.folder)
        results = src.get_all_documents_metadata(limit=2)

        self.assertEqual(len(results), 2)

    def test_get_all_documents_metadata_modified_is_utc_aware(self):
        """The ``modified`` field must be a timezone-aware UTC datetime."""
        self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        result = src.get_all_documents_metadata()[0]

        self.assertIsNotNone(result["modified"].tzinfo)

    # ---------------------------------------------------------------------- #
    # get_list_artefacts                                                      #
    # ---------------------------------------------------------------------- #

    def test_get_list_artefacts_returns_all_when_last_synced_is_none(self):
        """When last_synced is None, all PDFs in the folder must be returned.

        This mirrors the initial-sync behaviour: with no prior checkpoint every
        discovered file is treated as new.
        """
        self.write_dummy_pdf(self.folder, "doc1.pdf")
        self.write_dummy_pdf(self.folder, "doc2.pdf")

        src = self.connect_source(self.folder)
        artefacts = src.get_list_artefacts(None)

        self.assertEqual(len(artefacts), 2)

    def test_get_list_artefacts_returns_tuples_of_str_and_datetime(self):
        """Each artefact must be a (str, datetime) tuple with a stable string ID."""
        self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        artefacts = src.get_list_artefacts(None)

        self.assertIsInstance(artefacts[0][0], str)
        self.assertIsInstance(artefacts[0][1], datetime)

    def test_get_list_artefacts_ids_are_relative_paths(self):
        """Artefact IDs must be relative to folder_path, not absolute."""
        self.write_dummy_pdf(self.folder, "relative.pdf")
        src = self.connect_source(self.folder)
        artefact_id = src.get_list_artefacts(None)[0][0]

        self.assertFalse(artefact_id.startswith("/"))
        self.assertEqual(artefact_id, "relative.pdf")

    def test_get_list_artefacts_sorted_ascending_by_mtime(self):
        """Artefacts must be sorted by modification time, oldest first."""
        import time

        old = self.write_dummy_pdf(self.folder, "old.pdf")
        time.sleep(0.05)
        new = self.write_dummy_pdf(self.folder, "new.pdf")

        src = self.connect_source(self.folder)
        artefacts = src.get_list_artefacts(None)

        ids = [a[0] for a in artefacts]
        self.assertEqual(ids.index("old.pdf") < ids.index("new.pdf"), True)

    def test_get_list_artefacts_filters_by_last_synced(self):
        """Only files modified strictly after last_synced must be returned.

        This is the incremental-sync path: files that predate the checkpoint
        should be excluded so they are not reprocessed unnecessarily.
        """
        self.write_dummy_pdf(self.folder, "existing.pdf")

        future = datetime(9999, 1, 1, tzinfo=timezone.utc)
        src = self.connect_source(self.folder)
        artefacts = src.get_list_artefacts(future)

        self.assertEqual(len(artefacts), 0)

    def test_get_list_artefacts_timestamps_are_utc_aware(self):
        """Returned timestamps must be timezone-aware (UTC)."""
        self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        _, ts = src.get_list_artefacts(None)[0]

        self.assertIsNotNone(ts.tzinfo)

    def test_get_list_artefacts_scans_subdirectories(self):
        """PDFs in subdirectories must be discovered."""
        sub = self.folder / "sub"
        sub.mkdir()
        self.write_dummy_pdf(sub, "deep.pdf")

        src = self.connect_source(self.folder)
        ids = [a[0] for a in src.get_list_artefacts(None)]

        self.assertIn("sub/deep.pdf", ids)

    # ---------------------------------------------------------------------- #
    # get_content                                                             #
    # ---------------------------------------------------------------------- #

    def test_get_content_returns_one_content_per_artefact(self):
        """get_content() must return exactly one Content object per input artefact.

        This test verifies the basic contract of the method: every artefact that
        enters must produce a corresponding Content, even if Docling conversion
        fails.  The parser is mocked to return a well-formed ParsedDocument.
        """
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        artefacts = [(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))]
        contents = src.get_content(artefacts)

        self.assertEqual(len(contents), 1)

    def test_get_content_id_matches_artefact_key(self):
        """Content.id_ must equal the relative path provided in the artefact tuple."""
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        artefacts = [(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))]
        content = src.get_content(artefacts)[0]

        self.assertEqual(content.id_, pdf.name)

    def test_get_content_date_matches_artefact_timestamp(self):
        """Content.date must match the timestamp supplied in the artefact tuple.

        The date reflects the mtime observed during listing, not the time the
        content was retrieved, so that incremental sync cursors stay accurate.
        """
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        expected_date = datetime(2024, 6, 15, tzinfo=timezone.utc)
        content = src.get_content([(pdf.name, expected_date)])[0]

        self.assertEqual(content.date, expected_date)

    def test_get_content_payload_contains_required_keys(self):
        """Content.content must contain all mandatory payload keys."""
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]

        for key in ("file_path", "file_name", "file_size", "num_pages", "pdf_meta",
                    "num_sections", "num_tables", "num_figures", "section_titles", "chunks"):
            self.assertIn(key, content.content, f"Missing key: {key}")

    def test_get_content_chunks_produced_from_sections(self):
        """When sections are present, chunks must be produced by the chunking strategy.

        This test verifies the end-to-end wiring between DocumentParserDocling
        and the chunking strategy: sections from the ParsedDocument flow through
        SectionChunkingStrategy and end up serialised in Content.content["chunks"].
        """
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]

        self.assertGreater(len(content.content["chunks"]), 0)

    def test_get_content_chunks_are_serialised_dicts(self):
        """Chunks in Content.content must be plain dicts (model_dump output), not Chunk objects."""
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]
        chunks = content.content["chunks"]

        self.assertIsInstance(chunks[0], dict)
        self.assertIn("document_id", chunks[0])
        self.assertIn("chunk_index", chunks[0])
        self.assertIn("text", chunks[0])

    def test_get_content_calls_embedder_when_configured(self):
        """When an embedder is configured, embed() must be called with the produced chunks.

        This test verifies the wiring between the chunking step and the embedding
        step by checking that the embedder's embed() method is called with a
        non-empty list of Chunk objects.
        """
        pdf = self.write_dummy_pdf(self.folder)

        spy_embedder = SpyEmbedder()

        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(embedder=spy_embedder),
        })
        src._parser = MagicMock()
        src._parser.parse.return_value = self.make_parsed_document()

        src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])

        self.assertGreater(len(spy_embedder.received), 0, "embed() was never called")
        self.assertIsInstance(spy_embedder.received[0], Chunk)

    def test_get_content_skips_embedder_when_not_configured(self):
        """When no embedder is configured, chunks must have empty vectors.

        This verifies the default behaviour: embedding is opt-in and the
        pipeline must produce usable text-only chunks without an embedder.
        """
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document()

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]

        for chunk in content.content["chunks"]:
            self.assertEqual(chunk["vector"], [])

    def test_get_content_raises_key_error_for_missing_file(self):
        """get_content() must raise KeyError when an artefact path no longer exists.

        This guards against the race condition where a file is deleted between
        listing and retrieval.
        """
        src = self.connect_source(self.folder)
        artefacts = [("ghost.pdf", datetime(2024, 1, 1, tzinfo=timezone.utc))]

        with self.assertRaises(KeyError):
            src.get_content(artefacts)

    def test_get_content_continues_after_conversion_failure(self):
        """A DocumentConversionError on one file must not abort the whole batch.

        The pipeline must still return a Content object for the failed file,
        with empty chunks, so that a single bad PDF does not prevent the rest
        of the batch from being processed.
        """
        pdf_ok  = self.write_dummy_pdf(self.folder, "ok.pdf")
        pdf_bad = self.write_dummy_pdf(self.folder, "bad.pdf")

        src = self.connect_source(self.folder)

        from database_builder_libs.utility.extract.document_parser_docling import (
            ConversionFault, DocumentConversionError,
        )

        def parse_side_effect(path: str):
            if "bad" in path:
                raise DocumentConversionError(faults=[
                    ConversionFault(faults=[], hashvalue="x", path_file_document=Path(path))
                ])
            return self.make_parsed_document()

        src._parser.parse.side_effect = parse_side_effect

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        contents = src.get_content([(pdf_ok.name, ts), (pdf_bad.name, ts)])

        self.assertEqual(len(contents), 2)
        bad_content = next(c for c in contents if c.id_ == "bad.pdf")
        self.assertEqual(bad_content.content["chunks"], [])

    def test_get_content_structural_meta_reflects_parsed_document(self):
        """Structural metadata counts must match the sections/tables/figures in ParsedDocument."""
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)

        parsed = self.make_parsed_document(sections=[
            ("Introduction", "Intro text padded to be long enough.", []),
            ("Methods",      "Methods text padded to be long enough.", [MagicMock()]),
        ])
        parsed.tables  = [MagicMock(), MagicMock()]
        parsed.figures = [MagicMock()]
        src._parser.parse.return_value = parsed

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]

        self.assertEqual(content.content["num_sections"], 2)
        self.assertEqual(content.content["num_tables"],   2)
        self.assertEqual(content.content["num_figures"],  1)

    def test_get_content_section_titles_in_metadata(self):
        """section_titles must list non-empty section headers from the parsed document."""
        pdf = self.write_dummy_pdf(self.folder)
        src = self.connect_source(self.folder)
        src._parser.parse.return_value = self.make_parsed_document(sections=[
            ("Background", "Background text long enough to chunk.", []),
            ("",           "Untitled preamble text long enough.",   []),
        ])

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]

        self.assertIn("Background", content.content["section_titles"])
        self.assertNotIn("", content.content["section_titles"])

    def test_get_content_uses_abstract_section_as_summary(self):
        """A section titled 'abstract' must be forwarded as the summary to the chunking strategy.

        This test verifies that _find_summary() correctly identifies the abstract
        section and that it reaches the chunking strategy via the summary kwarg,
        enabling strategies like SummaryAndSectionsStrategy to prepend it.
        """
        pdf = self.write_dummy_pdf(self.folder)

        spy = SpyStrategy()

        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(chunking_strategy=spy),
        })
        src._parser = MagicMock()
        src._parser.parse.return_value = self.make_parsed_document(sections=[
            ("Abstract", "This is the abstract of the paper.", []),
            ("Methods",  "This is the methods section text.",  []),
        ])

        src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])

        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0]["summary"], "This is the abstract of the paper.")

    def test_get_content_no_summary_when_no_abstract_section(self):
        """summary kwarg must be None when no abstract/summary section is present."""
        pdf = self.write_dummy_pdf(self.folder)

        spy = SpyStrategy()

        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(chunking_strategy=spy),
        })
        src._parser = MagicMock()
        src._parser.parse.return_value = self.make_parsed_document(sections=[
            ("Introduction", "Introduction text long enough.", []),
            ("Conclusion",   "Conclusion text long enough.",   []),
        ])

        src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])

        self.assertEqual(len(spy.calls), 1)
        self.assertIsNone(spy.calls[0]["summary"])

    def test_get_content_sections_disabled_produces_no_chunks(self):
        """When sections.enabled is False, Content.content["chunks"] must be empty."""
        pdf = self.write_dummy_pdf(self.folder)

        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(enabled=False),
        })
        src._parser = MagicMock()
        src._parser.parse.return_value = self.make_parsed_document()

        content = src.get_content([(pdf.name, datetime(2024, 1, 1, tzinfo=timezone.utc))])[0]

        self.assertEqual(content.content["chunks"], [])


if __name__ == "__main__":
    unittest.main()