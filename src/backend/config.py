import os
from typing_extensions import Final

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))
project = os.getenv("project")

project_env = find_dotenv(f'.env.{project.lower()}')
if project is not None:
    load_dotenv(project_env, override=True)

class Settings:
    TYPEDB_URI: Final[str] = os.getenv("TYPEDB_URI")
    TYPEDB_DATABASE: Final[str] = os.getenv("TYPEDB_DB")


settings = Settings()
