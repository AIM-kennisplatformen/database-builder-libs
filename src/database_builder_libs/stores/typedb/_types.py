from typing import Iterator, Mapping, TypedDict
from typedb.driver import ConceptDocumentIterator, ConceptRowIterator, QueryAnswer


class RelationRef(TypedDict):
    entity_type: str
    key_attr: str
    key: str


class RelationData(TypedDict, total=False):
    type: str
    roles: dict[str, RelationRef]
    attributes: Mapping[str, object]


class EagerQueryAnswer:
    """
    Eagerly evaluates a TypeDB QueryAnswer to prevent 'concurrent transaction close' errors
    when evaluating iterator wrappers outside of the transaction block.
    """
    def __init__(self, answer: QueryAnswer):
        self._is_docs = answer.is_concept_documents()
        self._is_rows = answer.is_concept_rows()
        if self._is_docs:
            self._docs = list(answer.as_concept_documents())
        elif self._is_rows:
            self._rows = list(answer.as_concept_rows())
        else:
            self._answer = answer

    def as_concept_documents(self) -> ConceptDocumentIterator:
        if not self._is_docs:
            raise TypeError("Query did not return concept documents")
        return iter(self._docs)

    def as_concept_rows(self) -> ConceptRowIterator:
        if not self._is_rows:
            raise TypeError("Query did not return concept rows")
        return iter(self._rows)

    def as_raw(self) -> QueryAnswer:
        if not self._answer:
            raise TypeError("Query is already evaluated as documents or rows")
        return self._answer

    def __iter__(self) -> Iterator:
        if self._is_docs:
            return iter(self._docs)
        if self._is_rows:
            return iter(self._rows)
        return iter([])
