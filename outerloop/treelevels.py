"""
Take trees and describe them in terms of their levels.
"""

import collections


TypedLevel = collections.namedtuple("TypedLevel", ["values", "grouplengths"])



def parse_typed(types, tree):
    """
    Convert a tree literal to a list of levels, building in an assumption
    that every node within a level has the same type.

    In this context, a node's "type" consists of:
    - What type of value does it have, if any?
    - Is it a leaf node? (i.e. is it the last level?)

    Thus, all leaf nodes will occur at the same level.

    Each type can be:
    - a callable that takes a list of values for a level and returns an object
    - None, indicating that this level has no values, only children

    The tree is only parsed down to the depth specified by `types`. If the tree
    literal goes deeper than that, then the remainder of the tree is stored as
    an unparsed treelevels literal as part of the value of the final level.
    """
    if types[-1] is None:
        raise ValueError("Leaf nodes must have type")

    levels = []
    level_nodes_grouped = [[tree]]

    # Handle non-leaf levels
    for i_level, t in enumerate(types[:-1]):

        level_values_unparsed = []
        next_level_groups = []

        for group in level_nodes_grouped:
            if not isinstance(group, collections.abc.Sequence):
                raise ValueError(f"Expected sequence, got {group}")

            for node in group:
                if t is None:
                    children = node
                else:
                    if not isinstance(node, collections.abc.Sequence) \
                       or len(node) != 2:
                        raise ValueError(
                            f"Expected value, children pair, got {node}")

                    value, children = node
                    level_values_unparsed.append(value)
                next_level_groups.append(children)

        levels.append(
            TypedLevel(
                values=(None if t is None else t(level_values_unparsed)),
                grouplengths=[len(group)
                              for group in level_nodes_grouped])
        )
        level_nodes_grouped = next_level_groups

    # Handle leaf level
    t = types[-1]
    if t is None:
        raise ValueError("Leaf nodes must have type")
    levels.append(
        TypedLevel(
            values=t([value
                      for group in level_nodes_grouped
                      for value in group]),
            grouplengths = [len(group)
                            for group in level_nodes_grouped]
        )
    )

    return levels


def dict_with_key(key):
    """
    Example:

    Given
      t = "k1"
    and values
      (41, 11, 21),
    return
      {"k1": [41, 11, 21]}
    """
    def construct_from_level_values(value_by_node):
        return {
            key: value_by_node
        }
    return construct_from_level_values


def dict_with_keys(keys):
    """
    Example:

    Given
      keys = ("k1", "k2")
    and values
      ((41, 42), (11, 12), (21, 22)),
    return
      {"k1": [41, 11, 21], "k2": [42, 12, 22]}
    """
    def construct_from_level_values(values_by_node):
        return {
            k: [values[i]
                for values in values_by_node]
            for i, k in enumerate(keys)
        }
    return construct_from_level_values


def parse(header_by_level, tree, use_tuples_as_keys=False):
    """
    A special case of treelevels.typed where every type is a dict with a
    known set of keys, with the tree literal containing only the values.

    Each item in header_by_level is either a single dict key or a list of dict
    keys.

    If "keys" is a single key, the node's value should be a single value. If
    multiple keys, the node's value should be be a list of values. In either
    case, the return value's TypedLevel.values will have template
      {"k1": [node1_value1, node2_value1, ...],
       "k2": [node1_value2, node2_value2, ...],
       ...}
    """
    types = []
    for header in header_by_level:
        if header is None:
            types.append(None)
        elif (isinstance(header, list)
              or (isinstance(header, collections.abc.Sequence)
                  and not isinstance(header, str)
                  and not use_tuples_as_keys)):
            if len(header) == 0:
                types.append(None)
            else:
                types.append(dict_with_keys(header))
        else:
            # assume "header" is a single key
            types.append(dict_with_key(header))

    return parse_typed(types, tree)


__all__ = [
    "typed",
    "keyed",
]
