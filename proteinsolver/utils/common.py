from collections import namedtuple
from typing import Any, Mapping


def to_namedtuple(name: str, dictionary: Mapping[str, Any]) -> Any:
    """Convert a dictionary to a NamedTuple.

    Useful for generating inputs to functions that expect NamedTuples.

    Examples:
        >> to_namedtuple("PandasRow", {"a": 10, "b": 20})
        PandasRow(a=10, b=20)
    """
    return namedtuple(name, dictionary.keys())(**dictionary)
