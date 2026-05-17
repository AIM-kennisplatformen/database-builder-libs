import sqlite3
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from database_builder_libs.utility.sync._base import (
    AbstractSyncTracker,
    Artifact,
    ConflictItem,
)

SCHEMA_VERSION = 1
DEFAULT_DB_PATH = "partial_sync.db"


def _retry(max_attempts: int = 3, base_delay: float = 0.1) -> Callable:
    """Retry a database operation on sqlite3.OperationalError."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self: "SqliteSyncTracker", *args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(self, *args, **kwargs)
                except sqlite3.OperationalError as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "Database operation failed (attempt {}/{}): {}."
                            " Retrying in {:.1f}s...",
                            attempt,
                            max_attempts,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
            logger.error("Database operation failed after {} attempts", max_attempts)
            raise RuntimeError(
                f"Database operation failed after {max_attempts} attempts"
            ) from last_exc

        return wrapper

    return decorator


class SqliteSyncTracker(AbstractSyncTracker):
    """
    SQLite-backed implementation of AbstractSyncTracker.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file. Use ``:memory:`` for an
        in-memory database. Defaults to ``"partial_sync.db"``.
    table_sources : str
        Name of the sources table. Default ``"sources"``.
    table_artifacts : str
        Name of the artifacts table. Default ``"artifacts"``.
    timeout : float
        Connection timeout in seconds. Default ``5.0``.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        table_sources: str = "sources",
        table_artifacts: str = "artifacts",
        timeout: float = 5.0,
    ) -> None:
        self._db_path = str(db_path) if db_path is not None else DEFAULT_DB_PATH
        self._table_sources = table_sources
        self._table_artifacts = table_artifacts
        self._timeout = timeout

        self.conn = sqlite3.connect(self._db_path, timeout=self._timeout)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        """Create or migrate database tables to match SCHEMA_VERSION."""
        self._create_schema_version_table()
        current_version = self._get_schema_version()

        if current_version < SCHEMA_VERSION:
            logger.info(
                "Migrating sync database from schema v{} to v{}",
                current_version,
                SCHEMA_VERSION,
            )
            self._run_migrations(current_version)
            self._set_schema_version(SCHEMA_VERSION)
            self.conn.commit()

    def _create_schema_version_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_version(
                version INTEGER PRIMARY KEY
            )
            """
        )

    def _get_schema_version(self) -> int:
        cur = self.conn.execute("SELECT MAX(version) FROM _schema_version")
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else 0

    def _set_schema_version(self, version: int) -> None:
        self.conn.execute("INSERT INTO _schema_version(version) VALUES (?)", (version,))

    def _run_migrations(self, from_version: int) -> None:
        """Run sequential migrations from `from_version` to SCHEMA_VERSION."""
        if from_version < 1:
            self._migrate_v1()

    def _migrate_v1(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_sources}(
                source_name TEXT PRIMARY KEY,
                last_sync_time REAL
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_artifacts}(
                item_key TEXT,
                source_name TEXT,
                modified_time REAL,
                last_sync_time REAL,
                PRIMARY KEY(item_key, source_name)
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_artifacts}_item_key
            ON {self._table_artifacts}(item_key)
            """
        )

    @_retry()
    def start_sync(self, source_name: str) -> Optional[float]:
        """Return the last sync timestamp for the source."""
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT last_sync_time FROM {self._table_sources} WHERE source_name=?",
            (source_name,),
        )
        row = cur.fetchone()
        if row:
            logger.info("Source '{}' last synced at {}", source_name, row[0])
            return row[0]
        cur.execute(
            f"INSERT INTO {self._table_sources}(source_name,last_sync_time) "
            "VALUES (?,NULL)",
            (source_name,),
        )
        self.conn.commit()
        logger.info("Registered new source '{}'", source_name)
        return None

    @_retry()
    def finish_sync(
        self,
        source_name: str,
        artifacts: list[Artifact],
    ) -> list[ConflictItem]:
        """
        Insert artifacts reported by the source and return
        artifacts requiring reconciliation.
        """
        cur = self.conn.cursor()
        now = time.time()
        rows = [
            (item_key, source_name, modified_time.timestamp(), now)
            for item_key, modified_time in artifacts
        ]
        cur.executemany(
            f"""
            INSERT INTO {self._table_artifacts}
                (item_key,source_name,modified_time,last_sync_time)
            VALUES(?,?,?,?)
            ON CONFLICT(item_key,source_name)
            DO UPDATE SET
                modified_time=excluded.modified_time,
                last_sync_time=excluded.last_sync_time
            """,
            rows,
        )
        cur.execute(
            f"UPDATE {self._table_sources} SET last_sync_time=? WHERE source_name=?",
            (now, source_name),
        )
        self.conn.commit()
        conflicts = self._find_conflicts(source_name)
        if conflicts:
            logger.warning(
                "Found {} conflicting artifact(s) for source '{}': {}",
                len(conflicts),
                source_name,
                conflicts,
            )
        else:
            logger.info(
                "Finished sync for source '{}' ({} artifacts, no conflicts)",
                source_name,
                len(artifacts),
            )
        return conflicts

    @_retry()
    def cleanup_old_records(self, before: float) -> int:
        """
        Remove artifact records older than the given timestamp.

        Parameters
        ----------
        before : float
            Unix timestamp. Records with ``last_sync_time < before``
            are deleted.

        Returns
        -------
        int
            Number of deleted artifact rows.
        """
        cur = self.conn.cursor()
        cur.execute(
            f"DELETE FROM {self._table_artifacts} WHERE last_sync_time < ?",
            (before,),
        )
        self.conn.commit()
        logger.info("Cleaned up {} old artifact record(s)", cur.rowcount)
        return cur.rowcount

    @_retry()
    def sync_history(self, source_name: str) -> list[dict[str, Any]]:
        """
        Return sync history for a given source.

        Parameters
        ----------
        source_name : str
            Unique identifier for the data source.

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains the keys ``item_key``, ``source_name``,
            ``modified_time``, and ``last_sync_time``.
        """
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT item_key, source_name, modified_time, last_sync_time "
            f"FROM {self._table_artifacts} "
            "WHERE source_name=? "
            "ORDER BY last_sync_time DESC",
            (source_name,),
        )
        return [
            {
                "item_key": row[0],
                "source_name": row[1],
                "modified_time": row[2],
                "last_sync_time": row[3],
            }
            for row in cur.fetchall()
        ]

    @_retry()
    def sync_stats(self) -> dict[str, Any]:
        """
        Return aggregate statistics about the tracked sync state.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_sources``, ``total_artifacts``,
            ``sources_with_conflicts``.
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self._table_sources}")
        total_sources = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {self._table_artifacts}")
        total_artifacts = cur.fetchone()[0]
        cur.execute(
            f"""
            SELECT COUNT(DISTINCT a.source_name)
            FROM {self._table_artifacts} a
            JOIN {self._table_artifacts} b
              ON a.item_key = b.item_key
             AND a.source_name != b.source_name
             AND a.modified_time != b.modified_time
            """
        )
        sources_with_conflicts = cur.fetchone()[0]
        return {
            "total_sources": total_sources,
            "total_artifacts": total_artifacts,
            "sources_with_conflicts": sources_with_conflicts,
        }

    @_retry()
    def _find_conflicts(self, source_name: str) -> list[ConflictItem]:
        """Return item_keys where sources disagree on modification time."""
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT a.item_key
            FROM {self._table_artifacts} a
            JOIN {self._table_artifacts} b
              ON a.item_key = b.item_key
            WHERE a.source_name = ?
              AND b.source_name != a.source_name
              AND a.modified_time != b.modified_time
            """,
            (source_name,),
        )
        return [row[0] for row in cur.fetchall()]

    @_retry()
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.debug("Sync database connection closed")
