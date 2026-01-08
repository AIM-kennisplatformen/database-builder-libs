from abc import ABC


# 
class Datastore(ABC):  # noqa: B024
    """
    Abstract base class that should be implemented by datasources to save entities and relations to various datastores.
    Exists for the purpose of typehinting.
    """
