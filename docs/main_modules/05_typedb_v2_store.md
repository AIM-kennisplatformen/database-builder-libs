# TypeDB V2 store

## Overview

The TypeDB V2 store allows you to store nodes using the Abstract store API. It maps canonical Node objects onto a TypeDB schema, providing graph database capabilities with strong typing and complex relationship modeling.


### Configuration
Configuration of the typedb store happens using a dict containing an url, database and schema_path.

```python
config = {
    "uri": "localhost:1729",
    "database": "knowledge_base",
    "schema_path": "path/to/schema.tql"  # Optional
}
```

## Filter Logic and Practical Examples

The TypeDB V2 store uses a URL-query style filter syntax for retrieving and deleting nodes. This section explains how filters work with practical examples.

### Filter Syntax

The basic filter format follows URL query string conventions:

```
entity=<entity_type>&<attribute>=<value>&<attribute>=<value>&include=relations
```

### Practical Examples

#### Example 1: Single Attribute Filter
```python
# Find a person by email
filter = "entity=person&email=john.doe@example.com"
nodes = store.get_nodes(filter)

# Find documents by title
filter = "entity=document&title=Research Paper 2024"
nodes = store.get_nodes(filter)
```

#### Example 2: Multiple Attribute Filters
```python
# Find a person by email AND department
filter = "entity=person&email=jane@company.com&department=Engineering"
nodes = store.get_nodes(filter)

# Find documents by author AND status
filter = "entity=document&author=Dr. Smith&status=published"
nodes = store.get_nodes(filter)
```

#### Example 3: Including Relations
```python
# Get person with all their relations loaded
filter = "entity=person&email=john@example.com&include=relations"
nodes = store.get_nodes(filter)

# The returned node will have its relations populated:
# node.relations will contain RelationData objects
for node in nodes:
    print(f"Found {len(node.relations)} relations for {node.id}")
    for rel in node.relations:
        print(f"  - {rel['type']} relation with roles: {rel['roles']}")
```

#### Example 4: Retrieving All Nodes
```python
# Get all nodes in the database (use with caution on large datasets)
nodes = store.get_nodes(None)

# This is equivalent to:
nodes = store.get_nodes(filter=None)
```

### Filter Behavior and Edge Cases

#### Empty vs None Filter
```python
# None filter returns ALL nodes
nodes = store.get_nodes(None)  # Returns everything

# Empty string filter raises ValueError
nodes = store.get_nodes("")  # Raises ValueError: filter cannot be empty
```

#### Case Sensitivity
```python
# Filters are case-sensitive
filter1 = "entity=person&name=John Doe"
filter2 = "entity=person&name=john doe"
# These will return different results
```

#### Special Characters in Values
```python
# Values with special characters are handled automatically
filter = "entity=person&email=user+tag@example.com"
# The + character is preserved correctly
```

### Deletion Filters

The same filter syntax applies to `remove_nodes()`:

#### Safe Deletion (Single Entity)
```python
# Delete a specific person
filter = "entity=person&email=john@example.com"
deleted_count = store.remove_nodes(filter)
```

#### Bulk Deletion (Requires Permission)
```python
# Delete all draft documents (requires allow_multiple=True)
filter = "entity=document&status=draft"
deleted_count = store.remove_nodes(filter, allow_multiple=True)

# Attempting bulk deletion without permission raises ValueError
filter = "entity=document"  # No attribute filter
deleted_count = store.remove_nodes(filter)  # Raises ValueError
```


## Docstring
::: database_builder_libs.stores.typedb_v2.typedb_v2_store
    handler: python