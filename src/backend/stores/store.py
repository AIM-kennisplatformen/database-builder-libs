from abc import ABC, abstractmethod

class Datastore(ABC):

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def save(self, data: dict) -> None:
        pass
