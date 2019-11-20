import random
from typing import Tuple

import numpy as np
import torch
from scipy import sparse


def construct_solved_sudoku():
    """Return a solved sudoku puzzle."""
    while True:
        try:
            puzzle = np.zeros((9, 9), dtype=np.int64)
            rows = [set(range(1, 10)) for i in range(9)]
            columns = [set(range(1, 10)) for i in range(9)]
            squares = [set(range(1, 10)) for i in range(9)]
            for i in range(9):
                for j in range(9):
                    # Pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i] & columns[j] & squares[(i // 3) * 3 + j // 3]
                    if not choices:
                        raise ValueError
                    choice = random.choice(list(choices))

                    puzzle[i, j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i // 3) * 3 + j // 3].discard(choice)
            return puzzle
        except ValueError:
            pass


def gen_sudoku_graph_dense() -> np.ndarray:
    """Generate Sudoku constraint graph (dense)."""
    # Create connectivity matrix for each pixel in graph
    lst = []
    for i in range(9):
        for j in range(9):
            a = np.zeros((9, 9), dtype=np.int)
            i_div = i // 3
            j_div = j // 3
            a[i_div * 3 : (i_div + 1) * 3, j_div * 3 : (j_div + 1) * 3] = 1
            a[i, :] = 1
            a[:, j] = 1
            lst.append(a)
    # Combine into a single connectivity matrix
    adj_dense = np.empty((81, 81), dtype=np.int)
    for i, a in enumerate(lst):
        adj_dense[i, :] = a.reshape(-1)
    return adj_dense


def gen_sudoku_graph_featured() -> np.ndarray:
    """Generate Sudoku constraint graph (dense)."""
    # Create connectivity matrix for each pixel in graph
    lst = []
    for i in range(9):
        for j in range(9):
            a = np.zeros((9, 9, 3), dtype=np.float32)
            i_div = i // 3
            j_div = j // 3
            a[i_div * 3 : (i_div + 1) * 3, j_div * 3 : (j_div + 1) * 3, 0] = 1
            a[i, :, 1] = 1
            a[:, j, 2] = 1
            lst.append(a)
    # Combine into a single connectivity matrix
    adj_dense = np.empty((81, 81, 3), dtype=np.float32)
    for i, a in enumerate(lst):
        adj_dense[i, :, :] = a.reshape(81, 3)
    return adj_dense


def gen_sudoku_graph() -> Tuple[torch.tensor, torch.tensor]:
    """Generate Sudoku constraint graph (sparse)."""
    adj_dense = gen_sudoku_graph_dense()
    adj_coo = sparse.coo_matrix(adj_dense)
    indices = torch.stack([torch.from_numpy(adj_coo.row), torch.from_numpy(adj_coo.col)]).to(
        torch.long
    )
    values = torch.from_numpy(adj_coo.data)
    return indices, values


def sudoku_is_solved(values) -> bool:
    ref = np.arange(1, 10)
    mat = values.reshape(9, 9)
    for i in range(3):
        for j in range(3):
            v = np.sort(mat[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3], axis=None)
            if not (v == ref).all():
                return False
    for i in range(9):
        v = np.sort(mat[i, :])
        if not (v == ref).all():
            return False
    for j in range(9):
        v = np.sort(mat[:, j])
        if not (v == ref).all():
            return False
    return True
