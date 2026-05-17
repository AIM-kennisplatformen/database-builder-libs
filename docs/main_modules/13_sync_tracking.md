# Sync tracking

The sync tracking module lives in `database_builder_libs.utility.sync`.

It provides an abstract interface (`AbstractSyncTracker`) and a SQLite-backed implementation (`SqliteSyncTracker`) for tracking per-source synchronization state and detecting artifact modification conflicts.

## Pattern

The typical sync workflow has three steps:

1. **`start_sync(source_name)`** — retrieve the last known sync timestamp for a source (or `None` if first sync)
2. **Fetch artifacts** — pass the timestamp to the source (e.g., `ZoteroSource.get_list_artefacts`) to get only changed items
3. **`finish_sync(source_name, artifacts)`** — upsert artifact records, update the source timestamp, and return any conflicting item keys

```python
from database_builder_libs.utility.sync import SqliteSyncTracker

tracker = SqliteSyncTracker()

last_sync = tracker.start_sync("Zotero")
last_sync_dt = (
    datetime.fromtimestamp(last_sync) if last_sync is not None else None
)

artifacts = zot.get_list_artefacts(last_synced=last_sync_dt)
conflicts = tracker.finish_sync("Zotero", artifacts)

if conflicts:
    print(f"Conflicts detected: {conflicts}")

tracker.close()
```

## Database schema

The SQLite implementation uses two configurable tables:

### `sources` table (default name)

| Column | Type | Description |
|---|---|---|
| `source_name` | TEXT PRIMARY KEY | Unique source identifier |
| `last_sync_time` | REAL | Unix timestamp of last successful sync |

### `artifacts` table (default name)

| Column | Type | Description |
|---|---|---|
| `item_key` | TEXT | Artifact identifier |
| `source_name` | TEXT | Source that reported this artifact |
| `modified_time` | REAL | Artifact's modification time in the source |
| `last_sync_time` | REAL | When this artifact was last synchronized |

Primary key is `(item_key, source_name)`. An index on `item_key` supports conflict detection joins.

### `_schema_version` table

Internal table tracking the database schema version for migrations.

## SqliteSyncTracker

### Configuration

```python
SqliteSyncTracker(
    db_path="partial_sync.db",  # Path or ":memory:"
    table_sources="sources",     # Custom sources table name
    table_artifacts="artifacts", # Custom artifacts table name
    timeout=5.0,                 # SQLite connection timeout
)
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `start_sync(source_name)` | `float \| None` | Last sync timestamp or None for new sources |
| `finish_sync(source_name, artifacts)` | `list[str]` | Upserts artifacts, returns conflicting item keys |
| `cleanup_old_records(before)` | `int` | Deletes artifact records older than timestamp |
| `sync_history(source_name)` | `list[dict]` | All recorded artifacts for a source |
| `sync_stats()` | `dict` | Aggregate counts (sources, artifacts, conflicts) |
| `close()` | — | Closes the database connection |

### Conflict detection

A conflict occurs when two sources report the same artifact (`item_key`) with different `modified_time` values. This typically indicates the artifact was updated in one source but not in another.

The `finish_sync` method automatically checks for conflicts and returns the problematic item keys. After reconciliation (updating the conflicting source), the conflict is resolved.

### Error handling

Database operations are wrapped with automatic retry (3 attempts with exponential backoff) for `sqlite3.OperationalError` (e.g., database lock). All operations are logged via `loguru`.

## In-memory database

Use `:memory:` as the database path for testing:

```python
tracker = SqliteSyncTracker(db_path=":memory:")
```

## Custom database backends

Implement `AbstractSyncTracker` to support other backends:

```python
from database_builder_libs.utility.sync import AbstractSyncTracker, Artifact, ConflictItem

class PostgresSyncTracker(AbstractSyncTracker):
    def start_sync(self, source_name: str) -> float | None:
        ...

    def finish_sync(
        self, source_name: str, artifacts: list[Artifact]
    ) -> list[ConflictItem]:
        ...

    def close(self) -> None:
        ...
```

## Troubleshooting

| Problem | Likely cause | Solution |
|---|---|---|
| `sqlite3.OperationalError: database is locked` | Concurrent access | Increase `timeout` or reduce concurrent writers |
| `sqlite3.OperationalError: no such table` | Schema version mismatch | Delete the `.db` file and let it recreate |
| Unexpected conflicts | Sources reporting different timestamps | Verify source clocks are synchronized (UTC) |
