# ProteinSolver: Solving the Inverse Protein Folding Problem with Graph Neural Networks

[![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fmybinder%3AhTGKLsjmxRS8xNyHxRJB%40https%3A%2F%2Fgitlab.com%2Fostrokach%2Fproteinsolver.git/v0.1.1)
[![docs](https://img.shields.io/badge/docs-v0.1.1-blue.svg)](https://ostrokach.gitlab.io/proteinsolver/d49e067ac2d5496f8b58f007bc8bd88e/v0.1.1/)
[![conda](https://img.shields.io/conda/dn/ostrokach-forge/proteinsolver.svg)](https://anaconda.org/ostrokach-forge/proteinsolver/)
[![build status](https://gitlab.com/ostrokach/proteinsolver/badges/v0.1.1/build.svg)](https://gitlab.com/ostrokach/proteinsolver/commits/v0.1.1/)
[![coverage report](https://gitlab.com/ostrokach/proteinsolver/badges/v0.1.1/coverage.svg)](https://ostrokach.gitlab.io/proteinsolver/d49e067ac2d5496f8b58f007bc8bd88e/v0.1.1/htmlcov/)

## Description

ProteinSolver is a deep neural network which learns to solve (ill-defined) constraint satisfaction problems (CSPs) from training data.

## Demo notebooks

The following notebooks can be used to explore the basic functionality of `proteinsolver`.

| Notebook name           | MyBinder                                                                                                                                                                                                                                | Description                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `20_test_sudoku.ipynb`  | [![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fmybinder%3AhTGKLsjmxRS8xNyHxRJB%40https%3A%2F%2Fgitlab.com%2Fostrokach%2Fproteinsolver.git/v0.1.1?filepath=notebooks%2F20_test_sudoku.ipynb) | Test a network trained to solve Sudoku puzzles.                        |
| `20_test_protein.ipynb` | [![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fmybinder%3AhTGKLsjmxRS8xNyHxRJB%40https%3A%2F%2Fgitlab.com%2Fostrokach%2Fproteinsolver.git/v0.1.1?filepath=notebooks%2F20_test_protein.ipynb) | Test a network trained to predict the amino acid sequence of proteins. |

Other notebooks in the `notebooks/` directory show how to perform more extensive validations of the networks and how to train new networks.

## Installation

We recommend installing `proteinsolver` into a clean conda environment using the following command:

```bash
conda create -n proteinsolver -c pytorch -c conda-forge -c kimlab -c ostrokach-forge proteinsolver
conda activate proteinsolver
```

## Development

First, use `conda` to install `proteinsolver` into a new conda environment. This will also install all dependencies.

```bash
conda create -n proteinsolver -c pytorch -c conda-forge -c kimlab -c ostrokach-forge proteinsolver
conda activate proteinsolver
```

Second, run `pip install --editable .` inside the root directory of this package. This will force Python to use the development version of our code.

```bash
cd path/to/proteinsolver
pip install --editable .
```

## Environment variables

- `DATAPKG_DATA_DIR` - Location of training and validation data.

## References
