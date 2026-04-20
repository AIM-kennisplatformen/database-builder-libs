import json
import os
import unittest
from unittest.mock import MagicMock
import tempfile

import httpretty
from httpretty import HTTPretty
from loguru import logger
from datetime import datetime, timezone


from database_builder_libs.sources.zotero_source import ZoteroSource, FileType


class ZoteroTests(unittest.TestCase):
    """Tests for ZoteroSource"""

    cwd = os.path.dirname(os.path.realpath(__file__))
    def connect_source(self) -> ZoteroSource:
        src = ZoteroSource()
        src.connect(
            {
                "library_id": "myuserid",
                "library_type": "user",
                "api_key": "fakekey",
            }
        )
        return src

    def get_doc(self, doc_name):
        with open(
            os.path.join(self.cwd, "zotero_api_responses", doc_name)
        ) as f:
            return f.read()

    def setUp(self):
        self.items_doc = self.get_doc("items_doc.json")
        self.item_file = self.get_doc("item_file.pdf")

    @httpretty.activate
    def test_get_all_documents_metadata_http(self):
        """Test the `get_all_documents_metadata` method from zotero source using HTTP mocks

        This test verifies both the underlying pyzotero and zotero source methods by mocking the zotero API
        using HTTPretty. It provides some troubleshooting and regression insights beyond testing the functionality of methods.
        """
        zot = self.connect_source()

        HTTPretty.register_uri(
            HTTPretty.GET,
            "https://api.zotero.org/users/myuserid/collections/COLL123/items/top",
            content_type="application/json",
            body=self.items_doc,
        )

        result = zot.get_all_documents_metadata("COLL123")

        assert json.loads(self.items_doc) == result

    def test_download_zotero_item_calls_dump_with_attachment(self):
        """Test the `download_zotero_item` call using mock methods.

        This test verifies the outcome of the `download_zotero_item` method by mocking the pyzotero: `children` & `dump` api.
        Used to check for attachments and downloading attachments from cloud_api.
        It is expected that tested function calls children and checks for attachments and calls dump accordingly.
        It verifies the logic of the implemented `download_zotero_item` method.
        """
        fake_children = [
            {
                "key": "ATTACH123",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                },
            }
        ]

        fake_zotero = MagicMock()
        fake_zotero.children.return_value = fake_children

        source = self.connect_source()
        source._zotero = fake_zotero

        source.download_zotero_item(
            item_id="ITEM123",
            download_path="/tmp",
        )

        fake_zotero.children.assert_called_once_with("ITEM123")
        fake_zotero.dump.assert_called_once_with(
            itemkey="ATTACH123",
            filename="ITEM123.pdf",
            path="/tmp",
        )
    
    def test_download_zotero_item_with_accept_types_pdf_only(self):
        """Test download_zotero_item with accept_types=["pdf"] and allow_fallback=False"""
        fake_children = [
            {
                "key": "ATTACH_PDF",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                },
            },
            {
                "key": "ATTACH_EPUB",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/epub+zip",
                },
            }
        ]

        fake_zotero = MagicMock()
        fake_zotero.children.return_value = fake_children

        source = self.connect_source()
        source._zotero = fake_zotero

        # Should select PDF even though EPUB is also present
        source.download_zotero_item(
            item_id="ITEM123",
            download_path="/tmp",
            accept_types=["pdf"],
            allow_fallback=False,
        )

        # Verify PDF was selected (not EPUB)
        fake_zotero.dump.assert_called_once_with(
            itemkey="ATTACH_PDF",
            filename="ITEM123.pdf",
            path="/tmp",
        )

    def test_download_zotero_item_with_accept_types_epub_first(self):
        """Test download_zotero_item with accept_types=["epub", "pdf"] (EPUB preferred)"""
        fake_children = [
            {
                "key": "ATTACH_PDF",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                },
            },
            {
                "key": "ATTACH_EPUB",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/epub+zip",
                },
            }
        ]

        fake_zotero = MagicMock()
        fake_zotero.children.return_value = fake_children

        source = self.connect_source()
        source._zotero = fake_zotero

        # Should select EPUB (preferred) not PDF
        source.download_zotero_item(
            item_id="ITEM123",
            download_path="/tmp",
            accept_types=["epub", "pdf"],
        )

        # Verify EPUB was selected
        fake_zotero.dump.assert_called_once_with(
            itemkey="ATTACH_EPUB",
            filename="ITEM123.epub",
            path="/tmp",
        )

    def test_download_zotero_item_returns_false_when_no_acceptable_type(self):
        """Test that download_zotero_item returns False when no acceptable type found and allow_fallback=False"""
        fake_children = [
            {
                "key": "ATTACH_HTML",
                "data": {
                    "itemType": "attachment",
                    "contentType": "text/html",
                },
            }
        ]

        fake_zotero = MagicMock()
        fake_zotero.children.return_value = fake_children

        source = self.connect_source()
        source._zotero = fake_zotero

        # Request PDF only, but only HTML available, with no fallback
        result = source.download_zotero_item(
            item_id="ITEM123",
            download_path="/tmp",
            accept_types=["pdf"],
            allow_fallback=False,
        )

        # Should return False (no acceptable type)
        assert result is False
        # dump should NOT be called
        fake_zotero.dump.assert_not_called()

    def test_download_zotero_item_fallback_when_preferred_type_missing(self):
        """Test fallback behavior when preferred type not found but allow_fallback=True"""
        fake_children = [
            {
                "key": "ATTACH_EPUB",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/epub+zip",
                },
            }
        ]

        fake_zotero = MagicMock()
        fake_zotero.children.return_value = fake_children

        source = self.connect_source()
        source._zotero = fake_zotero

        # Request PDF with fallback enabled, but only EPUB available
        result = source.download_zotero_item(
            item_id="ITEM123",
            download_path="/tmp",
            accept_types=["pdf"],
            allow_fallback=True,  # Should fall back to smart selection
        )

        # Should succeed and use EPUB (via fallback)
        assert result is True
        fake_zotero.dump.assert_called_once()
        call_args = fake_zotero.dump.call_args
        assert call_args[1]["filename"] == "ITEM123.epub"

    def test_file_type_constants(self):
        """Test that FileType constants are correctly defined"""
        assert FileType.PDF == ["pdf"]
        assert FileType.EPUB == ["epub"]
        assert FileType.EBOOKS == ["epub", "pdf"]
        assert FileType.DOCUMENTS == ["pdf", "docx", "doc"]
        assert "pdf" in FileType.ALL
        assert "epub" in FileType.ALL

    def test_download_zotero_item_with_correct_file_extensions(self):
        """Test that different file types get correct extensions"""
        test_cases = [
            ("application/pdf", "ITEM.pdf"),
            ("application/epub+zip", "ITEM.epub"),
            ("application/msword", "ITEM.doc"),
            ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "ITEM.docx"),
            ("text/plain", "ITEM.txt"),
            ("text/html", "ITEM.html"),
        ]

        for content_type, expected_filename in test_cases:
            fake_children = [
                {
                    "key": "ATTACH",
                    "data": {
                        "itemType": "attachment",
                        "contentType": content_type,
                    },
                }
            ]

            fake_zotero = MagicMock()
            fake_zotero.children.return_value = fake_children

            source = self.connect_source()
            source._zotero = fake_zotero

            source.download_zotero_item(
                item_id="ITEM",
                download_path="/tmp",
            )

            # Verify correct filename with extension
            fake_zotero.dump.assert_called_once()
            call_args = fake_zotero.dump.call_args
            assert call_args[1]["filename"] == expected_filename

    @httpretty.activate
    def test_list_artefacts_str_ids_http(self):
        """Ensure Zotero item keys are returned as stable string identifiers"""

        zot = self.connect_source()
        BASE = "https://api.zotero.org/users/myuserid"

        # prevent accidental real HTTP
        httpretty.enable(allow_net_connect=False)

        HTTPretty.register_uri(
            HTTPretty.GET,
            f"{BASE}/items",
            content_type="application/json",
            adding_headers={"Link": '<>; rel="last"'},
            body=json.dumps(
                [
                    {
                        "key": "ABCD1234",
                        "data": {
                            "key": "ABCD1234",
                            "dateModified": "2024-05-10T12:30:00Z",
                        },
                    },
                    {
                        "key": "XYZ98765",
                        "data": {
                            "key": "XYZ98765",
                            "dateModified": "2023-01-01T00:00:00Z",
                        },
                    },
                ]
            ),
        )

        artefacts = zot.get_list_artefacts(None)
        # -----------------------
        # Assertions
        # -----------------------
        assert len(artefacts) == 2

        # IDs must be str, not UUID
        assert isinstance(artefacts[0][0], str)

        # exact Zotero keys preserved
        assert artefacts[0][0] == "ABCD1234"
        assert artefacts[1][0] == "XYZ98765"

        # datetime parsed correctly
        assert artefacts[0][1].isoformat() == "2024-05-10T12:30:00+00:00"
        assert artefacts[1][1].isoformat() == "2023-01-01T00:00:00+00:00"

    @httpretty.activate
    def test_download_zotero_item_http(self):
        """Test the `download_zotero_item` method from zotero source using HTTP mocks

        This test verifies both the underlying pyzotero and zotero source methods by mocking the zotero API
        using HTTPretty. It provides some troubleshooting and regression insights beyond testing the functionality of methods.
        """
        zot = self.connect_source()

        HTTPretty.register_uri(
            HTTPretty.GET,
            "https://api.zotero.org/users/myuserid/items/MYITEMID/children",
            content_type="application/json",
            body=json.dumps(
                [
                    {
                        "key": "ATTACH123",
                        "data": {
                            "itemType": "attachment",
                            "linkMode": "imported_file",
                            "contentType": "application/pdf",
                            "filename": "myitemid.pdf",
                        },
                    }
                ]
            ),
        )

        HTTPretty.register_uri(
            HTTPretty.GET,
            "https://api.zotero.org/users/myuserid/items/ATTACH123/file",
            content_type="application/pdf",
            body=self.item_file,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            zot.download_zotero_item(
                item_id="myitemid",
                download_path=tmpdir,
            )

            downloaded = os.path.join(tmpdir, "myitemid.pdf")
            assert os.path.exists(downloaded)

            with open(downloaded, "rb") as f:
                items_data = f.read()

            logger.debug("Downloaded PDF bytes: {}", items_data)
            assert b"One very strange PDF\n" == items_data

    def test_get_all_documents_metadata_returns_everything(self):
        """Test the `get_all_documents_metadata` call using mock methods.

        This test verifies the outcome of the `get_all_documents_metadata` method by mocking the pyzotero: `collection_items_top` & `everything` api.
        Everything is a method within pyzotero api that allows to chain calls for every item within collection.
        Collection_items_top is method to retrieve metadata of specific item in zotero.
        It is expected that tested method calls everything with collection_items_top nested within, to retrieve metadata of all documents in collection.

        This test thus verifies the logic of the implemented `get_all_documents_metadata` method.
        """
        expected_items = [
            {"key": "A1", "data": {"title": "Doc 1"}},
            {"key": "A2", "data": {"title": "Doc 2"}},
        ]

        fake_iterator = object()

        fake_zotero = MagicMock()
        fake_zotero.collection_items_top.return_value = fake_iterator
        fake_zotero.everything.return_value = expected_items

        source = self.connect_source()
        source._zotero = fake_zotero

        result = source.get_all_documents_metadata("COLL123")

        fake_zotero.collection_items_top.assert_called_once_with(
            "COLL123",
            limit=None,
        )
        fake_zotero.everything.assert_called_once_with(fake_iterator)
        assert result == expected_items

    @httpretty.activate
    def test_get_content_http(self):
        """Verify get_content fetches items using string Zotero IDs"""

        zot = self.connect_source()
        BASE = "https://api.zotero.org/users/myuserid"

        # Block real internet
        httpretty.enable(allow_net_connect=False)

        # ---------------------------
        # Mock item metadata
        # ---------------------------
        HTTPretty.register_uri(
            HTTPretty.GET,
            f"{BASE}/items/ABCD1234",
            content_type="application/json",
            body=json.dumps(
                {
                    "key": "ABCD1234",
                    "data": {
                        "key": "ABCD1234",
                        "title": "Test Paper",
                        "itemType": "journalArticle",
                    },
                }
            ),
        )

        HTTPretty.register_uri(
            HTTPretty.GET,
            f"{BASE}/items/XYZ98765",
            content_type="application/json",
            body=json.dumps(
                {
                    "key": "XYZ98765",
                    "data": {
                        "key": "XYZ98765",
                        "title": "Another Paper",
                        "itemType": "book",
                    },
                }
            ),
        )

        # ---------------------------
        # Input artefacts (from list step)
        # ---------------------------
        artefacts = [
            ("ABCD1234", datetime(2024, 5, 10, 12, 30, tzinfo=timezone.utc)),
            ("XYZ98765", datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]


        contents = zot.get_content(artefacts)

        # ---------------------------
        # Assertions
        # ---------------------------
        assert len(contents) == 2

        # IDs preserved exactly
        assert contents[0].id_ == "ABCD1234"
        assert contents[1].id_ == "XYZ98765"

        # Metadata retrieved
        assert contents[0].content["title"] == "Test Paper"
        assert contents[1].content["title"] == "Another Paper"

        # Date comes from artefacts (important!)
        assert contents[0].date == artefacts[0][1]
        assert contents[1].date == artefacts[1][1]

if __name__ == "__main__":
    unittest.main()