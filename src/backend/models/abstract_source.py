from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Content(BaseModel):
    date: datetime
    id_: str
    content: dict


class AbstractSource(ABC, BaseModel):
    """
    connects optionally to the source,
    retrieves data from the source,
    optionally preprocesses the data (eg pdf to text, metadata extraction, etc)
    """

    @abstractmethod
    def connect_to_source(self) -> None:
        """Abstract method to connect to the source."""
        ...

    @abstractmethod
    def get_list_artefacts(
        self, last_synced: Optional[datetime]
    ) -> list[tuple[str, datetime]]:
        """Abstract method to preprocess data from the source."""
        ...

    @abstractmethod
    def get_content(self, artefacts: list[tuple[str, datetime]]) -> list[Content]:
        """Abstract method to get data from the source."""
        ...

    def __call__(self, last_synced: Optional[datetime]) -> list[Content]:
        self.connect_to_source()
        artefacts = self.get_list_artefacts(last_synced)
        contents = self.get_content(artefacts)
        return contents
