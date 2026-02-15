"""
neo4j_graphql_shim.py

Drop-in replacement for `from neo4j_graphql_py import neo4j_graphql`.

Handles arbitrary nesting depth by recursively building OPTIONAL MATCH chains.
Each level of nesting appends its own OPTIONAL MATCH + WITH aggregation block.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from graphql import GraphQLResolveInfo, GraphQLList, GraphQLNonNull, GraphQLObjectType

logger = logging.getLogger("neo4j_graphql_py")


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _unwrap_type(t):
    while isinstance(t, (GraphQLNonNull, GraphQLList)):
        t = t.of_type
    return t

def _is_list_type(t):
    while isinstance(t, GraphQLNonNull):
        t = t.of_type
    return isinstance(t, GraphQLList)

def _get_relation_directive(field_def) -> Optional[Dict]:
    if field_def.ast_node is None:
        return None
    for directive in field_def.ast_node.directives:
        if directive.name.value == "relation":
            args = {arg.name.value: arg.value.value for arg in directive.arguments}
            return {
                "name": args.get("name", ""),
                "direction": args.get("direction", "OUT").upper(),
            }
    return None

def _selected_scalar_fields(selection_set, parent_type) -> List[str]:
    scalars = []
    if selection_set is None:
        return scalars
    for sel in selection_set.selections:
        fname = sel.name.value
        if fname.startswith("__"):
            continue
        fd = parent_type.fields.get(fname)
        if fd is None:
            continue
        if sel.selection_set is None:
            scalars.append(fname)
    return scalars

def _selected_relation_fields(selection_set, parent_type):
    if selection_set is None:
        return
    for sel in selection_set.selections:
        fname = sel.name.value
        if fname.startswith("__"):
            continue
        fd = parent_type.fields.get(fname)
        if fd is None:
            continue
        if sel.selection_set is None:
            continue
        rel = _get_relation_directive(fd)
        if rel is None:
            continue
        sub_kwargs = {
            arg.name.value: arg.value.value
            for arg in sel.arguments
            if hasattr(arg.value, "value")
        }
        yield fname, fd, rel, sel.selection_set, sub_kwargs

def _build_where(alias: str, filters: Dict) -> str:
    if not filters:
        return ""
    clauses = []
    for k, v in filters.items():
        clauses.append(f'{alias}.{k} = "{v}"' if isinstance(v, str) else f"{alias}.{k} = {v}")
    return " AND ".join(clauses)


# ---------------------------------------------------------------------------
# Recursive subquery builder
# ---------------------------------------------------------------------------

def _build_subquery(
    parent_alias: str,
    node_type: GraphQLObjectType,
    selection_set,
    rel_name: str,
    direction: str,
    sub_kwargs: Dict,
    is_list: bool,
    carried: List[str],       # variable names already in scope (for WITH carry-forward)
    parts: List[str],         # append Cypher lines here (mutates in place)
    alias_counter: List[int], # mutable counter for unique aliases
) -> str:
    """
    Appends OPTIONAL MATCH + recursive WITH blocks to `parts`.
    Returns the final alias name for this subquery's collected result.
    """
    alias_counter[0] += 1
    node_alias = f"{node_type.name.lower()}_{alias_counter[0]}"
    label = node_type.name

    if direction == "OUT":
        pattern = f"({parent_alias})-[:{rel_name}]->({node_alias}:{label})"
    else:
        pattern = f"({parent_alias})<-[:{rel_name}]-({node_alias}:{label})"

    parts.append(f"OPTIONAL MATCH {pattern}")

    sub_where = _build_where(node_alias, sub_kwargs)
    if sub_where:
        parts.append(f"WHERE {sub_where}")
        # OPTIONAL MATCH + WHERE only nullifies non-matching rows; it does NOT
        # exclude them from the result set.  Emitting a WITH...WHERE IS NOT NULL
        # immediately after turns it into a real filter so parent rows that have
        # no qualifying children are dropped before we recurse / aggregate.
        filter_with = list(carried) + [node_alias]
        parts.append(f"WITH {', '.join(filter_with)}")
        parts.append(f"WHERE {node_alias} IS NOT NULL")

    # Scalars on this node
    scalars = _selected_scalar_fields(selection_set, node_type)

    # Recurse into nested relation fields
    nested_carried = list(carried) + [node_alias]
    nested_aliases: List[Tuple[str, str]] = []  # (field_name, collected_alias)

    for fname, fd, rel, sub_sel, skwargs in _selected_relation_fields(selection_set, node_type):
        related_type = _unwrap_type(fd.type)
        if not isinstance(related_type, GraphQLObjectType):
            continue
        collected_alias = _build_subquery(
            parent_alias=node_alias,
            node_type=related_type,
            selection_set=sub_sel,
            rel_name=rel["name"],
            direction=rel["direction"],
            sub_kwargs=skwargs,
            is_list=_is_list_type(fd.type),
            carried=nested_carried,
            parts=parts,
            alias_counter=alias_counter,
        )
        nested_carried.append(collected_alias)
        nested_aliases.append((fname, collected_alias))

    # Build the map expression for this node
    map_parts = [f"{f}: {node_alias}.{f}" for f in scalars]
    map_parts += [f"{fname}: {alias}" for fname, alias in nested_aliases]
    map_expr = f"{{ {', '.join(map_parts)} }}" if map_parts else node_alias

    # Collect or single based on cardinality
    if is_list:
        collect_expr = f"collect({map_expr})"
    else:
        collect_expr = map_expr  # singular — just a map, no collect

    # Unique alias for this collected result
    result_alias = f"{node_alias}_collected"

    # WITH to carry everything forward and aggregate this level
    with_vars = list(carried) + [f"{collect_expr} AS {result_alias}"]
    parts.append(f"WITH {', '.join(with_vars)}")

    return result_alias


# ---------------------------------------------------------------------------
# Top-level Cypher builder
# ---------------------------------------------------------------------------

def _build_cypher(return_type: GraphQLObjectType, selection_set, kwargs: Dict) -> str:
    label = return_type.name
    root_alias = return_type.name.lower()

    scalars = _selected_scalar_fields(selection_set, return_type)
    relation_fields = list(_selected_relation_fields(selection_set, return_type))

    parts = [f"MATCH ({root_alias}:{label})"]
    where = _build_where(root_alias, kwargs)
    if where:
        parts.append(f"WHERE {where}")

    if not relation_fields:
        if scalars:
            scalar_map = ", ".join(f"{f}: {root_alias}.{f}" for f in scalars)
            parts.append(f"RETURN {{ {scalar_map} }} AS result")
        else:
            parts.append(f"RETURN {root_alias} AS result")
    else:
        parts.append(f"WITH {root_alias}")

        alias_counter = [0]
        carried = [root_alias]
        top_aliases: List[Tuple[str, str]] = []  # (field_name, collected_alias)

        for fname, fd, rel, sub_sel, skwargs in relation_fields:
            related_type = _unwrap_type(fd.type)
            if not isinstance(related_type, GraphQLObjectType):
                continue
            collected_alias = _build_subquery(
                parent_alias=root_alias,
                node_type=related_type,
                selection_set=sub_sel,
                rel_name=rel["name"],
                direction=rel["direction"],
                sub_kwargs=skwargs,
                is_list=_is_list_type(fd.type),
                carried=carried,
                parts=parts,
                alias_counter=alias_counter,
            )
            carried.append(collected_alias)
            top_aliases.append((fname, collected_alias))

        # Final RETURN map
        return_parts = [f"{f}: {root_alias}.{f}" for f in scalars]
        return_parts += [f"{fname}: {alias}" for fname, alias in top_aliases]
        parts.append(f"RETURN {{ {', '.join(return_parts)} }} AS result")

    cypher = "\n".join(parts)
    logger.debug("Generated Cypher:\n%s", cypher)
    return cypher


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def neo4j_graphql(obj: Any, context: Dict, info: GraphQLResolveInfo, **kwargs) -> Any:
    """Drop-in replacement for neo4j_graphql_py.neo4j_graphql()."""
    driver = context.get("driver")
    if driver is None:
        raise ValueError("neo4j_graphql_shim: context must contain 'driver'")

    field_def = info.parent_type.fields[info.field_name]
    return_type = _unwrap_type(field_def.type)

    if not isinstance(return_type, GraphQLObjectType):
        raise TypeError(
            f"neo4j_graphql_shim: field '{info.field_name}' does not return "
            f"an ObjectType (got {return_type})"
        )

    cypher = _build_cypher(
        return_type=return_type,
        selection_set=info.field_nodes[0].selection_set,
        kwargs=kwargs,
    )

    is_list = _is_list_type(field_def.type)

    with driver.session() as session:
        result = session.run(cypher)
        records = [record["result"] for record in result]

    logger.debug("Query returned %d record(s)", len(records))

    return records if is_list else (records[0] if records else None)