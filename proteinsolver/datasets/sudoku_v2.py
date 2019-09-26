from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset

from proteinsolver.datasets import download_url
from proteinsolver.utils import construct_solved_sudoku, gen_sudoku_graph, str_to_tensor


class SudokuDataset2(InMemoryDataset):
    def __init__(
        self,
        root,
        subset: Optional[str] = None,
        data_url: Optional[str] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        """Create new SudokuDataset."""
        self.data_url = (
            f"https://storage.googleapis.com/deep-protein-gen/sudoku/sudoku_{subset}.csv.gz"
            if data_url is None
            else data_url
        )
        self._raw_file_names = [self.data_url.rsplit("/")[-1]]
        self._edge_index, _ = gen_sudoku_graph()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return self._raw_file_names

    def download(self):
        download_url(self.data_url, self.raw_dir)

    def process(self):
        df = pd.read_csv(self.raw_paths[0], index_col=False)

        data_list = []
        for tup in df.itertuples():
            solution = str_to_tensor(tup.solutions) - 1
            if hasattr(tup, "quizzes"):
                quiz = str_to_tensor(tup.quizzes) - 1
                quiz = torch.where(quiz >= 0, quiz, torch.tensor(9))
                data = Data(x=quiz, y=solution)
            else:
                data = Data(x=solution)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super().get(idx)
        data.edge_index = self._edge_index
        return data


class SudokuDataset3(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None) -> None:
        self._gen_puzzle = construct_solved_sudoku
        self._edge_index, _ = gen_sudoku_graph()
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def __len__(self):
        return 700_000  # To be consistent with SudokuDataset2

    def get(self, idx):
        puzzle = torch.from_numpy(self._gen_puzzle().reshape(-1) - 1)
        data = Data(x=puzzle, edge_index=self._edge_index)
        return data
