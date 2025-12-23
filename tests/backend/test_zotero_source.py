import json
import os
import unittest
from unittest.mock import MagicMock
import tempfile

import httpretty
from httpretty import HTTPretty
from loguru import logger

from backend.sources.zotero_source import ZoteroSource


class ZoteroTests(unittest.TestCase):
    """Tests for ZoteroSource"""

    cwd = os.path.dirname(os.path.realpath(__file__))

    def get_doc(self, doc_name):
        with open(os.path.join(self.cwd, "web_api_mocks/zotero_api_responses", doc_name)) as f:
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
        zot = ZoteroSource("myuserid", "user", "myuserkey")

        HTTPretty.register_uri(
            HTTPretty.GET,
            "https://api.zotero.org/users/myuserid/collections/COLL123/items/top",
            content_type="application/json",
            body=self.items_doc,
        )

        result = zot.get_all_documents_metadata("COLL123")

        self.assertEqual(json.loads(self.items_doc), result)

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

        source = ZoteroSource.__new__(ZoteroSource)
        source.zotero = fake_zotero

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

    @httpretty.activate
    def test_download_zotero_item_http(self):
        """Test the `download_zotero_item` method from zotero source using HTTP mocks

            This test verifies both the underlying pyzotero and zotero source methods by mocking the zotero API
            using HTTPretty. It provides some troubleshooting and regression insights beyond testing the functionality of methods.
        """
        zot = ZoteroSource("myuserid", "user", "myuserkey")

        HTTPretty.register_uri(
            HTTPretty.GET,
            "https://api.zotero.org/users/myuserid/items/MYITEMID/children",
            content_type="application/json",
            body=json.dumps([
                {
                    "key": "ATTACH123",
                    "data": {
                        "itemType": "attachment",
                        "linkMode": "imported_file",
                        "contentType": "application/pdf",
                        "filename": "myitemid.pdf",
                    },
                }
            ]),
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
            self.assertTrue(os.path.exists(downloaded))

            with open(downloaded, "rb") as f:
                items_data = f.read()

            logger.debug("Downloaded PDF bytes: {}", items_data)
            self.assertEqual(b"One very strange PDF\n", items_data)

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

        source = ZoteroSource.__new__(ZoteroSource)
        source.zotero = fake_zotero

        result = source.get_all_documents_metadata("COLL123")

        fake_zotero.collection_items_top.assert_called_once_with(
            "COLL123",
            limit=None,
        )
        fake_zotero.everything.assert_called_once_with(fake_iterator)
        self.assertEqual(result, expected_items)


if __name__ == "__main__":
    unittest.main()
