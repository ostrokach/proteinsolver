import concurrent.futures
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

NOTEBOOK_NAME = "generate_difficult_sudokus"


NOTEBOOK_PATH = Path(NOTEBOOK_NAME).resolve()
NOTEBOOK_PATH.mkdir(exist_ok=True)
print(NOTEBOOK_PATH)


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


def decode_sugen_output(output):
    grid = np.empty((9, 9), dtype=np.int)
    for i, row in enumerate(output.split("\n")[:9]):
        for j, value in enumerate(row.split(" ")):
            if value == "_":
                grid[i, j] = 0
            else:
                grid[i, j] = int(value)
    return grid


def generate_sudoku():
    sc = "sugen -i 2000 -t 10000 generate"
    ps = subprocess.run(shlex.split(sc), stdout=subprocess.PIPE)

    sc2 = "sugen solve"
    ps2 = subprocess.run(shlex.split(sc2), input=ps.stdout, stdout=subprocess.PIPE)

    puzzle = decode_sugen_output(ps.stdout.decode())
    solution = decode_sugen_output(ps2.stdout.decode())

    assert sudoku_is_solved(solution)
    return puzzle, solution


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name")
    args = parser.parse_args()

    print(generate_sudoku())

    batch_idx = -1
    while True:
        batch_idx += 1
        print(f"Generating batch {batch_idx}...")
        results = []
        with concurrent.futures.ProcessPoolExecutor(psutil.cpu_count(logical=True)) as pool:
            futures = [pool.submit(generate_sudoku) for _ in range(100_000)]
            for future in concurrent.futures.as_completed(futures):
                puzzle, solution = future.result()
                puzzle_str = "".join([str(v) for v in puzzle.reshape(-1)])
                solution_str = "".join([str(v) for v in solution.reshape(-1)])
                results.append((puzzle_str, solution_str))
        df = pd.DataFrame(results, columns=["quizzes", "solutions"])
        df.to_csv(
            NOTEBOOK_PATH.joinpath(f"sodoku_{args.node_name}_{batch_idx}.csv"), sep=",", index=False
        )
