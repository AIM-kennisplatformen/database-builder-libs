from typing import Iterator, Mapping, TypedDict
from typedb.driver import QueryAnswer


class RelationRef(TypedDict):
    """
    Reference to a specific entity playing a role in a relation.
    
    Attributes
    ----------
    entity_type : str
        The type name of the entity.
    key_attr : str
        The name of the attribute serving as the key in the TypeDB schema.
    key : str
        The string value of the key attribute.
    """
    entity_type: str
    key_attr: str
    key: str


class RelationData(TypedDict, total=False):
    """
    Representation of a TypeDB relation.
    
    Attributes
    ----------
    type : str
        The relation type name.
    roles : dict[str, RelationRef]
        A mapping of role names to the entities (RelationRef) playing those roles.
    attributes : Mapping[str, object]
        Optional mapping of attribute names to their values for this relation.
    """
    type: str
    roles: dict[str, RelationRef]
    attributes: Mapping[str, object]


class EagerQueryAnswer:
    """
    Eagerly evaluates a TypeDB QueryAnswer to prevent 'concurrent transaction close' errors
    when evaluating iterator wrappers outside of the transaction block.
    """
    is_docs: bool
    is_rows: bool
    _docs: list
    _rows: list
    _answer: QueryAnswer | None

    def __init__(self, answer: QueryAnswer):
        """Initialize the EagerQueryAnswer by eagerly consuming the result iterator."""
        self._is_docs = answer.is_concept_documents()
        self._is_rows = answer.is_concept_rows()
        if self._is_docs:
            self._docs = list(answer.as_concept_documents())
        elif self._is_rows:
            self._rows = list(answer.as_concept_rows())
        else:
            self._answer = answer

    def as_concept_documents(self) -> Iterator:
        """Return the pre-evaluated documents iterator."""
        if not self._is_docs:
            raise TypeError("Query did not return concept documents")
        # This is put from an ConceptDocumentIterator to a list and then into a generic Iterator in order to move the data out of the transaction
        # Returned as an iterator for consistency with how the raw Query Answer is used
        return iter(self._docs)

    def as_concept_rows(self) -> Iterator:
        """Return the pre-evaluated rows iterator."""
        if not self._is_rows:
            raise TypeError("Query did not return concept rows")
        return iter(self._rows)

    def as_raw(self) -> QueryAnswer:
        """Return the underlying raw typeDB.QueryAnswer object, if not evaluated yet."""
        if not self._answer:
            raise TypeError("Query is already evaluated as documents or rows")
        # This is put from an ConceptDocumentIterator to a list and then into a generic Iterator in order to move the data out of the transaction
        # Returned as an iterator for consistency with how the raw Query Answer is used
        return self._answer

    def __iter__(self) -> Iterator:
        """Yield items from the eagerly evaluated result."""
        if self._is_docs:
            return iter(self._docs)
        if self._is_rows:
            return iter(self._rows)
        return iter([])
