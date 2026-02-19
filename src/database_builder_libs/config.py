import os
from typing_extensions import Final

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
project = os.getenv("project")

if project is not None:
    project_env = find_dotenv(f".env.{project.lower()}")
    load_dotenv(project_env, override=True)


class Settings:
    QDRANT_URL: Final[str] = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: Final[str] = os.getenv("QDRANT_COLLECTION", "knowledge_platform")
    QDRANT_VECTOR_SIZE: Final[int] = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

    TYPEDB_URI: Final[str] = os.getenv("TYPEDB_URI", "localhost:1729")
    TYPEDB_DATABASE: Final[str] = os.getenv("TYPEDB_DB", "knowledge_platform")

    ZOTERO_LIBRARY_ID: Final[str] = os.getenv("ZOTERO_LIBRARY_ID", "myuserid")
    ZOTERO_LIBRARY_TYPE: Final[str] = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    ZOTERO_API_KEY: Final[str] = os.getenv("ZOTERO_API_KEY", "myuserkey")


settings = Settings()
