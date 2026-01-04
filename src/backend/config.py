import os
from typing_extensions import Final

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))
project = os.getenv("project")

project_env = find_dotenv(f'.env.{project.lower()}')
if project is not None:
    load_dotenv(project_env, override=True)

class Settings:
    QDRANT_URL: Final[str] = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION: Final[str] = os.getenv("QDRANT_COLLECTION")
    QDRANT_VECTOR_SIZE: Final[int] = int(os.getenv("QDRANT_VECTOR_SIZE"))


settings = Settings()