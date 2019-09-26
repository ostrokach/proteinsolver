from pathlib import Path
from typing import List

import pytest
from ruamel import yaml

from proteinsolver.utils import str_to_tensor, sudoku_is_solved


def read_config(test_name: str) -> List:
    py_file = Path(__file__).resolve()
    yaml_file = py_file.with_suffix(".yaml")
    with yaml_file.open("rt") as fin:
        test_data = yaml.safe_load(fin.read())
    return test_data[test_name]


def parametrize(name, params_string):
    params = [s.strip() for s in params_string.split(",")]
    data_dicts = read_config(name)
    data_tuples = [tuple(d[p] for p in params) for d in data_dicts]
    return pytest.mark.parametrize(params_string, data_tuples)


@parametrize("test_sudoku_is_solved", "values, is_solved")
def test_sudoku_is_solved(values, is_solved):
    t = str_to_tensor(values).data.numpy()
    is_solved_ = sudoku_is_solved(t)
    assert is_solved_ == is_solved
