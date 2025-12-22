from abc import ABC, abstractmethod

from dataclasses import dataclass
from functools import singledispatchmethod

from backend.stores.qdrant_store import QdrantDatastore
from backend.stores.typedb_store import TypeDbDatastore
from backend.stores.store import Datastore

@dataclass
class AbstractSink(ABC):
    type: str
    hash_value: str

    @abstractmethod
    @singledispatchmethod
    def save_entity(self, store: "Datastore") -> None: ...

    @abstractmethod
    @save_entity.register
    def _(self, store: "QdrantDatastore") -> None: ...

    @abstractmethod
    @save_entity.register
    def _(self, store: "TypeDbDatastore") -> None: ...

    @abstractmethod
    @singledispatchmethod
    def save_relation(self, store: "Datastore") -> None: ...

    @abstractmethod
    @save_relation.register
    def _(self, store: "TypeDbDatastore") -> None: ...
