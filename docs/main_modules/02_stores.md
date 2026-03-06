# Abstract store

## Overview

The abstract store is the interface which all database adapters share. It provides a unified API for persisting and retrieving Node objects across different backend implementations (SQL, NoSQL, Graph databases, etc.).

## Design Notes

### Interaction Pattern

The AbstractStore follows these interaction patterns:

1. **Storage Pattern**:
    - Check if node exists (by id)
    - Update if exists, insert if new
    - Maintain idempotency

2. **Retrieval Pattern**:
    - String filter: Return direct matches preserving multiplicity
    - None filter: Return canonical, deduplicated node set
    - Stable ordering for identical queries

3. **Deletion Pattern**:
    - Single node removal with safety checks
    - Must fail on ambiguous matches

## Docstring abstract node

::: database_builder_libs.models.node
    handler: python

## Docstring abstract store

::: database_builder_libs.models.abstract_store
    handler: python


