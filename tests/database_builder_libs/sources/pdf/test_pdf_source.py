import dataclasses
import os
import time
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from database_builder_libs.models.abstract_chunk_embedder import AbstractChunkEmbedder
from database_builder_libs.models.abstract_chunk_strategy import AbstractChunkingStrategy, RawSection
from database_builder_libs.models.chunk import Chunk
from database_builder_libs.sources.pdf_source import (
    Acknowledgement,
    DocumentMetadata,
    ExtractionStrategy,
    FieldExtractionConfig,
    Institution,
    OrderedStrategyConfig,
    PDFDocumentConfig,
    PDFSource,
    SectionsConfig,
)
from database_builder_libs.utility.extract.document_parser_docling import (
    ConversionFault,
    DocumentConversionError,
)
from pydantic import Field


# --------------------------------------------------------------------------- #
# Helpers / spies                                                              #
# --------------------------------------------------------------------------- #

class SpyStrategy(AbstractChunkingStrategy):
    """Records chunk() calls; returns empty list."""
    def __init__(self):
        self.calls: list[dict] = []

    def chunk(self, sections: Sequence[RawSection], *, document_id: str,
              summary: str | None = None) -> list[Chunk]:
        self.calls.append({"sections": sections, "document_id": document_id, "summary": summary})
        return []


class SpyEmbedder(AbstractChunkEmbedder):
    """Records embed() calls; returns chunks with a fixed vector."""
    received: list[Chunk] = Field(default_factory=list)
    model_config = {"arbitrary_types_allowed": True}

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        self.received = list(chunks)
        return [
            Chunk(document_id=c.document_id, chunk_index=c.chunk_index,
                  text=c.text, vector=[0.1, 0.2, 0.3], metadata=c.metadata)
            for c in chunks
        ]


def _make_parsed(sections=None):
    """Return a ParsedDocument-shaped mock."""
    parsed = MagicMock()
    parsed.sections = sections or [
        ("Introduction", "This is a long enough introduction section.", []),
        ("Methods",      "The methods section describes the approach taken.", []),
        ("Abstract",     "This paper presents findings on fuel poverty.", []),
    ]
    parsed.tables  = []
    parsed.figures = []
    parsed.doc.pages = [MagicMock()]
    parsed.doc.iterate_items.return_value = iter([])
    return parsed


def _stub_parser(src: PDFSource, parsed=None, fail=False):
    """Wire src._parser to return *parsed* or raise DocumentConversionError."""
    src._parser = MagicMock()
    if fail:
        src._parser.parse.side_effect = DocumentConversionError(faults=[
            ConversionFault(faults=[], hashvalue="x", path_file_document=Path("bad.pdf"))
        ])
    else:
        src._parser.parse.return_value = parsed or _make_parsed()


class PDFSourceTests(unittest.TestCase):
    """Tests for PDFSource"""

    # ------------------------------------------------------------------ #
    # Fixtures                                                             #
    # ------------------------------------------------------------------ #

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.folder  = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def connect(self, extra: dict | None = None) -> PDFSource:
        """Connect a PDFSource with the Docling parser mocked out."""
        src = PDFSource()
        src.connect({"folder_path": str(self.folder), **(extra or {})})
        _stub_parser(src)
        return src

    def write_pdf(self, name: str = "paper.pdf") -> Path:
        p = self.folder / name
        p.write_bytes(b"")
        return p

    def ts(self) -> datetime:
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

    # ------------------------------------------------------------------ #
    # connect / lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def test_connect_succeeds_with_valid_folder(self):
        """connect() must not raise for an existing directory."""
        src = PDFSource()
        src.connect({"folder_path": str(self.folder)})

    def test_connect_raises_on_missing_folder(self):
        """connect() must raise ValueError when folder_path does not exist."""
        src = PDFSource()
        with self.assertRaises(ValueError):
            src.connect({"folder_path": "/does/not/exist"})

    def test_connect_is_idempotent(self):
        """Calling connect() twice must not re-initialise the parser."""
        src = PDFSource()
        src.connect({"folder_path": str(self.folder)})
        parser_id = id(src._parser)
        src.connect({"folder_path": str(self.folder)})
        self.assertEqual(id(src._parser), parser_id)

    def test_methods_raise_before_connect(self):
        """All public methods must raise RuntimeError before connect()."""
        src = PDFSource()
        with self.assertRaises(RuntimeError):
            src.get_list_artefacts(None)
        with self.assertRaises(RuntimeError):
            src.get_content([])
        with self.assertRaises(RuntimeError):
            src.get_all_documents_metadata()

    # ------------------------------------------------------------------ #
    # get_list_artefacts                                                   #
    # ------------------------------------------------------------------ #

    def test_list_returns_all_when_last_synced_is_none(self):
        """None last_synced must return every PDF discovered."""
        self.write_pdf("a.pdf")
        self.write_pdf("b.pdf")
        src = self.connect()
        self.assertEqual(len(src.get_list_artefacts(None)), 2)

    def test_list_ids_are_relative_paths(self):
        """Artefact IDs must be relative to folder_path."""
        sub = self.folder / "sub"
        sub.mkdir()
        (sub / "deep.pdf").write_bytes(b"")
        src = self.connect()
        ids = [a[0] for a in src.get_list_artefacts(None)]
        self.assertIn("sub/deep.pdf", ids)
        self.assertFalse(any(i.startswith("/") for i in ids))

    def test_list_sorted_ascending_by_mtime(self):
        """Artefacts must be ordered oldest-first."""
        old = self.write_pdf("old.pdf")
        time.sleep(0.05)
        self.write_pdf("new.pdf")
        src = self.connect()
        ids = [a[0] for a in src.get_list_artefacts(None)]
        self.assertLess(ids.index("old.pdf"), ids.index("new.pdf"))

    def test_list_filters_by_last_synced(self):
        """Files older than last_synced must be excluded."""
        self.write_pdf()
        src = self.connect()
        future = datetime(9999, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(src.get_list_artefacts(future), [])

    def test_list_timestamps_are_utc_aware(self):
        self.write_pdf()
        src = self.connect()
        _, ts = src.get_list_artefacts(None)[0]
        self.assertIsNotNone(ts.tzinfo)

    # ------------------------------------------------------------------ #
    # get_content — pipeline wiring                                        #
    # ------------------------------------------------------------------ #

    def test_get_content_returns_one_per_artefact(self):
        """One Content object must be returned per input artefact."""
        pdf = self.write_pdf()
        src = self.connect()
        contents = src.get_content([(pdf.name, self.ts())])
        self.assertEqual(len(contents), 1)

    def test_get_content_id_and_date(self):
        """Content.id_ and Content.date must match the artefact tuple."""
        pdf = self.write_pdf()
        src = self.connect()
        ts  = self.ts()
        c   = src.get_content([(pdf.name, ts)])[0]
        self.assertEqual(c.id_,  pdf.name)
        self.assertEqual(c.date, ts)

    def test_get_content_payload_keys(self):
        """Content.content must contain all required keys."""
        pdf = self.write_pdf()
        src = self.connect()
        payload = src.get_content([(pdf.name, self.ts())])[0].content
        for key in ("file_path", "file_name", "file_size", "num_pages",
                    "pdf_meta", "metadata", "chunks",
                    "num_sections", "num_tables", "num_figures", "section_titles"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_get_content_raises_key_error_for_missing_file(self):
        """KeyError must be raised when an artefact file no longer exists."""
        src = self.connect()
        with self.assertRaises(KeyError):
            src.get_content([("ghost.pdf", self.ts())])

    def test_get_content_continues_after_conversion_failure(self):
        """A conversion failure on one file must not abort the batch."""
        ok_pdf  = self.write_pdf("ok.pdf")
        bad_pdf = self.write_pdf("bad.pdf")

        src = PDFSource()
        src.connect({"folder_path": str(self.folder)})

        def side_effect(path: str):
            if "bad" in path:
                raise DocumentConversionError(faults=[
                    ConversionFault(faults=[], hashvalue="x",
                                    path_file_document=Path(path))
                ])
            return _make_parsed()

        src._parser = MagicMock()
        src._parser.parse.side_effect = side_effect

        ts       = self.ts()
        contents = src.get_content([(ok_pdf.name, ts), (bad_pdf.name, ts)])
        self.assertEqual(len(contents), 2)
        bad = next(c for c in contents if c.id_ == "bad.pdf")
        self.assertEqual(bad.content["chunks"], [])

    def test_get_content_chunks_are_dicts(self):
        """Chunks in Content.content must be plain dicts (asdict output)."""
        pdf = self.write_pdf()
        spy = SpyStrategy()
        spy.calls  # pre-touch
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(
                chunking_strategy=spy,
            ),
        })
        # Give the spy something to produce
        src._parser = MagicMock()
        parsed = _make_parsed()
        src._parser.parse.return_value = parsed
        # Override spy to return real chunks
        spy.chunk = lambda sections, *, document_id, summary=None: [
            Chunk(document_id=document_id, chunk_index=0,
                  text="hello world text", vector=[])
        ]
        payload = src.get_content([(self.write_pdf().name, self.ts())])[0].content
        self.assertIsInstance(payload["chunks"][0], dict)
        self.assertIn("document_id", payload["chunks"][0])

    def test_sections_disabled_produces_no_chunks(self):
        """sections.enabled=False must result in an empty chunks list."""
        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(enabled=False),
        })
        _stub_parser(src)
        payload = src.get_content([(pdf.name, self.ts())])[0].content
        self.assertEqual(payload["chunks"], [])

    def test_embedder_called_when_configured(self):
        """The configured embedder must receive the produced chunks."""
        spy_embedder = SpyEmbedder()

        # Use a strategy that actually produces chunks
        class OneChunkStrategy(AbstractChunkingStrategy):
            def chunk(self, sections, *, document_id, summary=None):
                return [Chunk(document_id=document_id, chunk_index=0,
                              text="section body text here", vector=[])]

        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(
                chunking_strategy=OneChunkStrategy(),
                embedder=spy_embedder,
            ),
        })
        _stub_parser(src)
        src.get_content([(pdf.name, self.ts())])
        self.assertGreater(len(spy_embedder.received), 0)
        self.assertIsInstance(spy_embedder.received[0], Chunk)

    def test_embedder_not_called_when_none(self):
        """Chunks must have empty vectors when no embedder is configured."""
        class OneChunkStrategy(AbstractChunkingStrategy):
            def chunk(self, sections, *, document_id, summary=None):
                return [Chunk(document_id=document_id, chunk_index=0,
                              text="section body text here", vector=[])]

        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "sections": SectionsConfig(chunking_strategy=OneChunkStrategy()),
        })
        _stub_parser(src)
        chunks = src.get_content([(pdf.name, self.ts())])[0].content["chunks"]
        self.assertTrue(all(c["vector"] == [] for c in chunks))

    # ------------------------------------------------------------------ #
    # Metadata — strategy dispatch                                         #
    # ------------------------------------------------------------------ #

    def _src_with_llm(self, llm_payload: dict, **extra_config) -> tuple[PDFSource, Path]:
        """
        Return a connected PDFSource whose LLM client returns *llm_payload*,
        together with a dummy PDF path.
        """
        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({"folder_path": str(self.folder), **extra_config})
        _stub_parser(src)
        src._llm_client = MagicMock()
        src._llm_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=__import__("json").dumps(llm_payload)
            ))]
        )
        return src, pdf

    def test_title_llm_strategy(self):
        """LLM strategy must populate title from llm_result['title']."""
        src, pdf = self._src_with_llm(
            {"title": "The Real Title", "authors": [], "acknowledgements": []},
            title=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
        )
        meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]
        self.assertEqual(meta["title"], "The Real Title")
        self.assertEqual(meta["source"]["title"], "llm")

    def test_title_docling_strategy(self):
        """DOCLING strategy must find title from the first section header."""
        from unittest.mock import patch as _patch
        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "title": FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.DOCLING])
            ),
        })
        _stub_parser(src)

        with _patch.object(PDFSource, "_first_section_header", return_value="Section Title Here"):
            meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]

        self.assertEqual(meta["title"], "Section Title Here")
        self.assertEqual(meta["source"]["title"], "docling_heuristic")

    def test_authors_llm_overwrites_file_metadata(self):
        """With stop_on_success=False, LLM must overwrite FILE_METADATA authors."""
        src, pdf = self._src_with_llm(
            {"authors": ["Real Author"], "acknowledgements": []},
            authors=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(
                    order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.LLM],
                    stop_on_success=False,
                )
            ),
        )
        fake_info = MagicMock()
        fake_info.get.side_effect = lambda key, *_: "Tool User" if key in ("/Author", "Author") else None
        fake_info.author = "Tool User"
        with patch("database_builder_libs.sources.pdf_source.PdfReader") as mock_reader:
            mock_reader.return_value.metadata = fake_info
            meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]

        self.assertEqual(meta["authors"], ["Real Author"])
        self.assertEqual(meta["source"]["authors"], "llm")

    def test_authors_stop_on_success_prevents_llm(self):
        """With stop_on_success=True, a successful FILE_METADATA must block LLM."""
        src, pdf = self._src_with_llm(
            {"authors": ["LLM Author"], "acknowledgements": []},
            authors=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(
                    order=[ExtractionStrategy.FILE_METADATA, ExtractionStrategy.LLM],
                    stop_on_success=True,
                )
            ),
        )
        # Patch PdfReader so pdf_info() returns a real author via FILE_METADATA.
        fake_info = MagicMock()
        fake_info.get.side_effect = lambda key, *_: "File Author" if key in ("/Author", "Author") else None
        fake_info.author = "File Author"
        with patch("database_builder_libs.sources.pdf_source.PdfReader") as mock_reader:
            mock_reader.return_value.metadata = fake_info
            meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]

        self.assertEqual(meta["authors"], ["File Author"])
        self.assertEqual(meta["source"]["authors"], "pdf_metadata")

    def test_summary_docling_strategy(self):
        """DOCLING summary strategy must pick up the abstract section body."""
        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "summary": FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.DOCLING])
            ),
        })
        parsed = _make_parsed(sections=[
            ("Abstract", "This is the abstract text of the document.", []),
            ("Methods",  "Method details here.", []),
        ])
        _stub_parser(src, parsed=parsed)

        with patch.object(PDFSource, "_find_summary",
                          return_value="This is the abstract text of the document."):
            meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]

        self.assertEqual(meta["summary"], "This is the abstract text of the document.")
        self.assertEqual(meta["source"]["summary"], "docling_heuristic")

    def test_publishing_institute_llm_strategy(self):
        """LLM strategy must populate publishing_institute."""
        src, pdf = self._src_with_llm(
            {"publishing_institute": "University Press", "authors": [], "acknowledgements": []},
            publishing_institute=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
        )
        meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]
        self.assertEqual(meta["publishing_institute"]["name"], "University Press")
        self.assertEqual(meta["source"]["publishing_institute"], "llm")

    def test_acknowledgements_llm_strategy(self):
        """LLM strategy must populate acknowledgements list."""
        src, pdf = self._src_with_llm(
            {
                "authors": [],
                "acknowledgements": [
                    {"name": "NEARG", "type": "organization", "relation": "collaboration"}
                ],
            },
            acknowledgements=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
        )
        meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]
        self.assertEqual(len(meta["acknowledgements"]), 1)
        self.assertEqual(meta["acknowledgements"][0]["name"], "NEARG")
        self.assertEqual(meta["source"]["acknowledgements"], "llm")

    def test_field_disabled_skips_extraction(self):
        """A field with enabled=False must produce no value and no source entry."""
        src, pdf = self._src_with_llm(
            {"title": "Should Not Appear", "authors": [], "acknowledgements": []},
            title=FieldExtractionConfig(enabled=False),
        )
        meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]
        self.assertIsNone(meta["title"])
        self.assertNotIn("title", meta["source"])

    def test_llm_not_called_when_not_configured(self):
        """LLM call must not be made when llm_base_url/llm_api_key are absent."""
        pdf = self.write_pdf()
        src = PDFSource()
        src.connect({
            "folder_path": str(self.folder),
            "authors": FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
        })
        _stub_parser(src)
        # No llm_client set — _call_llm must never be reached.
        self.assertIsNone(src._llm_client)
        meta = src.get_content([(pdf.name, self.ts())])[0].content["metadata"]
        self.assertIsNone(meta["authors"])

    def test_llm_called_once_even_for_multiple_fields(self):
        """The LLM must be invoked at most once per document regardless of how many fields need it."""
        src, pdf = self._src_with_llm(
            {"title": "T", "authors": ["A"], "publishing_institute": "P",
             "acknowledgements": []},
            title=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
            authors=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
            publishing_institute=FieldExtractionConfig(
                strategies=OrderedStrategyConfig(order=[ExtractionStrategy.LLM])
            ),
        )
        src.get_content([(pdf.name, self.ts())])
        self.assertEqual(
            src._llm_client.chat.completions.create.call_count, 1,
            "LLM was called more than once for a single document",
        )

    # ------------------------------------------------------------------ #
    # Metadata — helpers                                                   #
    # ------------------------------------------------------------------ #

    def test_split_authors_semicolon(self):
        self.assertEqual(
            PDFSource._split_authors("Smith J.; Doe A."),
            ["Smith J.", "Doe A."],
        )

    def test_split_authors_and(self):
        self.assertEqual(
            PDFSource._split_authors("Smith J. and Doe A."),
            ["Smith J.", "Doe A."],
        )

    def test_clean_meta_string_rejects_word_noise(self):
        self.assertIsNone(PDFSource._clean_meta_string("Microsoft Word - document.doc"))

    def test_clean_meta_string_accepts_real_title(self):
        self.assertEqual(
            PDFSource._clean_meta_string("  A Real Title  "),
            "A Real Title",
        )

    def test_clean_meta_string_rejects_junk(self):
        for junk in ("unknown", "untitled", "author"):
            self.assertIsNone(PDFSource._clean_meta_string(junk))

    def test_parse_author_line_abbreviated(self):
        result = PDFSource._parse_author_line("Smith J., Doe A.")
        self.assertEqual(result, ["J Smith", "A Doe"])

    def test_parse_author_line_no_match(self):
        self.assertEqual(PDFSource._parse_author_line("not an author line"), [])

    # ------------------------------------------------------------------ #
    # DocumentMetadata                                                     #
    # ------------------------------------------------------------------ #

    def test_document_metadata_default_fields(self):
        """All optional fields must default to None / empty."""
        meta = DocumentMetadata()
        self.assertIsNone(meta.title)
        self.assertIsNone(meta.authors)
        self.assertIsNone(meta.summary)
        self.assertIsNone(meta.publishing_institute)
        self.assertEqual(meta.acknowledgements, [])
        self.assertEqual(meta.source, {})
        self.assertIsNone(meta.keywords)

    def test_document_metadata_asdict_is_serialisable(self):
        """dataclasses.asdict() on DocumentMetadata must produce a JSON-serialisable dict."""
        import json
        meta = DocumentMetadata(
            title="Test",
            authors=["Author A"],
            publishing_institute=Institution(name="Press"),
            acknowledgements=[Acknowledgement(name="Group", type="organization", relation="funding")],
            source={"title": "llm"},
        )
        d = dataclasses.asdict(meta)
        json.dumps(d)  # must not raise


if __name__ == "__main__":
    unittest.main()