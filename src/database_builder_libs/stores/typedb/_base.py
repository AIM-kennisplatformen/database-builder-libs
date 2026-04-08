from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from loguru import logger
from typedb.driver import (
    Credentials,
    Driver,
    DriverOptions,
    Transaction,
    TransactionType,
    TypeDB,
)

from database_builder_libs.models.abstract_store import AbstractStore
from database_builder_libs.stores.typedb._types import EagerQueryAnswer


class TypeDbBase(AbstractStore):
    def __init__(self) -> None:
        """Initialize the base TypeDB driver state."""
        super().__init__()
        self.typedb_driver: Driver | None = None
        self.database: str | None = None
        self._entity_attr_cache: dict[str, list[str]] = {}
        self._all_attr_cache: list[str] | None = None
        self._key_attr_cache: dict[str, str | None] = {}

    def _ensure_connected(self) -> None:
        """Ensure that the driver and database are initialized."""
        super()._ensure_connected()
        assert self.typedb_driver is not None
        assert self.database is not None

    def _connect_impl(self, config: dict | None) -> None:
        """
        Connect to the TypeDB database using the provided configuration.
        Creates the database if missing and applies the schema if provided.
        """
        if not config:
            raise ValueError("TypeDB requires configuration")

        uri = config.get("uri")
        database = config.get("database")
        schema_path = config.get("schema_path")

        if not uri or not database:
            raise ValueError("TypeDB config requires 'uri' and 'database'")

        credentials = Credentials(str(config.get("username", "")), str(config.get("password", "")))
        driver_options = DriverOptions(is_tls_enabled=bool(config.get("tls", False)))

        self.typedb_driver = TypeDB.driver(address=uri, credentials=credentials, driver_options=driver_options)
        self.database = database

        # create database if missing
        if not self.typedb_driver.databases.contains(database):
            self.typedb_driver.databases.create(database)

        # apply schema if provided
        if schema_path:
            path = Path(schema_path)
            with self.transaction(TransactionType.SCHEMA) as tx:
                tx.query(path.read_text(encoding="utf-8")).resolve()
                tx.commit()

            # required by TypeDB after schema change
            self.typedb_driver.close()
            self.typedb_driver = TypeDB.driver(address=uri, credentials=credentials, driver_options=driver_options)

        for type_name, key_attr in self._load_key_attrs_from_schema().items():
            self._key_attr_cache[type_name] = key_attr

    @contextmanager
    def transaction(self, transaction_type: TransactionType) -> Generator[Transaction, None, None]:
        """
        Context manager for yielding a TypeDB transaction.
        Handles commits automatically for write and schema transactions.
        """
        self._ensure_connected()
        assert self.typedb_driver is not None
        assert self.database is not None

        with self.typedb_driver.transaction(database_name=self.database, transaction_type=transaction_type) as transaction:
            try:
                yield transaction
            except Exception:
                logger.warning("TypeDB transaction has encountered an error")
                # Do NOT commit — let TypeDB abort on close
                raise
            else:
                if (transaction_type.is_write() or transaction_type.is_schema()) and transaction.is_open():
                    transaction.commit()

    def query_read(self, query: str) -> EagerQueryAnswer:
        """Execute a read query and eagerly return its evaluated answer."""
        with self.transaction(TransactionType.READ) as tx:
            return EagerQueryAnswer(tx.query(query).resolve())

    def query_write(self, query: str) -> EagerQueryAnswer:
        """Execute a write query and eagerly return its evaluated answer."""
        with self.transaction(TransactionType.WRITE) as tx:
            return EagerQueryAnswer(tx.query(query).resolve())

    def query_schema(self, query: str) -> EagerQueryAnswer:
        """Execute a schema query, eagerly return its answer, and invalidate key caches."""
        with self.transaction(TransactionType.SCHEMA) as tx:
            result = EagerQueryAnswer(tx.query(query).resolve())
        # invalidate key attr cache — schema may have changed
        self._key_attr_cache.clear()
        self._all_attr_cache = None
        self._entity_attr_cache.clear()
        for type_name, key_attr in self._load_key_attrs_from_schema().items():
            self._key_attr_cache[type_name] = key_attr
        return result

    def _load_key_attrs_from_schema(self) -> dict[str, str]:
        """Load key attributes mapped by entity type from the database schema."""
        raise NotImplementedError("This method should be implemented by TypeDbSchemaMixin")
