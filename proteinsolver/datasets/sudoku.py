from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from proteinsolver.utils import gen_sudoku_graph


def str_to_tensor(s: str) -> torch.Tensor:
    t = torch.tensor([int(i) for i in s], dtype=torch.long)
    return t


class SudokuDataset(Dataset):
    def __init__(self, category: str = None) -> None:
        """Create new SudokuDataset.

        Args:
            category: One of {"train", "valid", "test"}, specifying which data subset to load.
        """
        assert category in ["train", "valid", "test"]
        df = pd.read_csv(
            Path(__file__).parent.joinpath("data", f"sudoku_{category}.csv.gz"), index_col=False
        )
        assert "quizzes" in df and "solutions" in df
        self._df = df
        self._edge_index, _ = gen_sudoku_graph()

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx) -> Data:
        quiz = str_to_tensor(self._df.at[idx, "quizzes"])
        solution = str_to_tensor(self._df.at[idx, "solutions"])
        data = Data(
            x=quiz.unsqueeze(dim=1), y=solution.unsqueeze(dim=1), edge_index=self._edge_index
        )
        data.label = torch.ones([1], dtype=torch.long)
        return data
