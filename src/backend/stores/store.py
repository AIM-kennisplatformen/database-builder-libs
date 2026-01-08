from abc import ABC

class Datastore(ABC):
    """
    Abstract base class that should be implemented by datasources to save entities and relations to various datastores.
    Exists for the purpose of typehinting.
    """