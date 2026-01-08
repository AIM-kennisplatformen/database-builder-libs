from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final
from typedb.driver import SessionType, TransactionType, TypeDB, TypeDBDriver, TypeDBOptions, TypeDBSession, TypeDBTransaction

from backend.config import settings
from backend.stores.store import Datastore

class TypeDbDatastore(Datastore):
    """
    TypeDbDatastore is a concrete implementation of the Datastore abstract base class for
    interacting with a TypeDB database.

    Data in TypeDB can be inserted, queried, and managed using TypeDB's schema and query language.

    Due to the implementation of the typedb-driver for TypeDB 2.x this datastore needs separate 
    methods for inserting, fetching, deleting, and updating data.
    """
    def __init__(self) -> None:
        self.typedb_driver: Final[TypeDBDriver] = TypeDB.core_driver(address=settings.TYPEDB_URI)
        self.database: Final[str] = settings.TYPEDB_DATABASE

        assert self.typedb_driver is not None, "TypeDB driver is not initialized."
        assert self.database is not None, "TypeDB database name is not set."

        if not self.typedb_driver.databases.contains(self.database):
            self.typedb_driver.databases.create(self.database)

        current_dir = Path(__file__).parent
        schema_path = current_dir / 'schema.tql'

        with self._query(SessionType.SCHEMA, TransactionType.WRITE) as transaction:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = f.read()
                transaction.query.define(schema)

    @contextmanager
    def _query(self, session_type: SessionType, transaction_type: TransactionType):
        session: TypeDBSession
        transaction: TypeDBTransaction
        with self.typedb_driver.session(self.database, session_type) as session:
            with session.transaction(transaction_type) as transaction:
                try:
                    yield transaction
                finally:
                    if transaction.is_open() and transaction.transaction_type.is_write():
                        transaction.commit()

    def save(self, query: str, options: TypeDBOptions = None) -> None:
        with self._query(SessionType.DATA, TransactionType.WRITE) as transaction:
            transaction.query.insert(query, options)
            transaction.commit()

    def delete(self, query: str, options: TypeDBOptions = None) -> None:
        with self._query(SessionType.DATA, TransactionType.WRITE) as transaction:
            transaction.query.delete(query, options)
            transaction.commit()

    def fetch(self, query: str, options: TypeDBOptions = None) -> list[Any]:
        with self._query(SessionType.DATA, TransactionType.READ) as transaction:
            iterator = transaction.query.fetch(query, options)
            results = [result.map() for result in iterator]
            return results

    def get(self, query: str, options: TypeDBOptions = None) -> list[Any]:
        with self._query(SessionType.DATA, TransactionType.READ) as transaction:
            iterator = transaction.query.get(query, options)
            results = [result.map() for result in iterator]
            return results

    def update(self, query: str, options: TypeDBOptions = None) -> None:
        with self._query(SessionType.DATA, TransactionType.WRITE) as transaction:
            transaction.query.update(query, options)
            transaction.commit()