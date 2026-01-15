from abc import ABC, abstractmethod

from functools import singledispatchmethod

from backend.stores.qdrant_store import QdrantDatastore
from backend.stores.typedb_store import TypeDbDatastore
from backend.stores.store import Datastore


class AbstractSink(ABC):
    """
    Abstract base class that should be implemented by datasources to save entities and relations to various datastores.
    """

    type: str

    @singledispatchmethod
    def save_entity(self, entity: dict, store: Datastore) -> None:
        """
        Save an entity to the specified datastore.
        This is an overloaded method that routes to type-specific implementations based on the datastore type.
        Args:
            entity (dict): The entity data to save. The dict should be structured according to the source.
            store (Datastore): The datastore instance where the entity will be saved.
        Raises:
            NotImplementedError: If no implementation exists for the given store type.
        """
        raise NotImplementedError(f"No implementation for {type(store)}")

    @save_entity.register
    def _(self, entity: dict, store: QdrantDatastore) -> None:
        self._save_qdrant_entity(entity, store)

    @abstractmethod
    def _save_qdrant_entity(self, entity: dict, store: QdrantDatastore) -> None: ...

    @save_entity.register
    def _(self, entity: dict, store: TypeDbDatastore) -> None:
        self._save_typedb_entity(entity, store)

    @abstractmethod
    def _save_typedb_entity(self, entity: dict, store: TypeDbDatastore) -> None: ...

    @singledispatchmethod
    def save_relation(self, relation: dict, store: Datastore) -> None:
        """
        Save a relation to the specified datastore.
        This is an overloaded method that routes to type-specific implementations based on the datastore type.
        Args:
            relation (dict): The relation data to save. The dict should be structured according to the source.
            store (Datastore): The datastore instance where the relation will be saved.
        Raises:
            NotImplementedError: If no implementation exists for the given store type.
        """
        raise NotImplementedError(f"No implementation for {type(store)}")

    @save_relation.register
    def _(self, relation: dict, store: TypeDbDatastore) -> None:
        self._save_typedb_relation(relation, store)

    @abstractmethod
    def _save_typedb_relation(self, relation: dict, store: TypeDbDatastore) -> None: ...
