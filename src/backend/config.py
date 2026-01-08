import os
from typing_extensions import Final

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))
project = os.getenv("project")

if project is not None:
    project_env = find_dotenv(f'.env.{project.lower()}')
    load_dotenv(project_env, override=True)

class Settings:
    TYPEDB_URI: Final[str] = os.getenv("TYPEDB_URI", "localhost:1729")
    TYPEDB_DATABASE: Final[str] = os.getenv("TYPEDB_DB", "knowledge_platform")


settings = Settings()
