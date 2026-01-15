from abc import ABC
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Content(BaseModel):
    date: datetime
    id_: UUID
    content: dict


class UUIDdata(BaseModel):
    def __call__(self, artefacts_ref: list[str]) -> list[UUID]:
        pass

    def check_uuid(self, artefact_ref: str) -> UUID:
        """checks in the database if an artefact is already present"""
        pass


class AbstractSource(ABC, BaseModel):
    """
    connects optionally to the source,
    retrieves data from the source,
    optionally preprocesses the data (eg pdf to text, metadata extraction, etc)
    """

    uuididata: UUIDdata = UUIDdata()

    def connect_to_source(self) -> None:
        """Abstract method to connect to the source."""
        pass

    def get_list_artefacts(
        self, last_synced: Optional[datetime]
    ) -> list[tuple[UUID, datetime]]:
        """Abstract method to preprocess data from the source."""
        pass

    def get_content(self, artefacts: list[UUID, datetime]) -> list[Content]:
        """Abstract method to get data from the source."""
        pass

    def __call__(self, last_synced: Optional[datetime]) -> list[Content]:
        self.connect_to_source()
        artefacts = self.get_list_artefacts(last_synced)
        contents = self.get_content(artefacts)
        return contents
