from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset

from proteinsolver import settings
from proteinsolver.datasets import download_url
from proteinsolver.utils import (
    construct_solved_sudoku,
    gen_sudoku_graph,
    gen_sudoku_graph_featured,
    str_to_tensor,
)


class SudokuDataset4(Dataset):
    def __init__(
        self, root, subset=None, data_url=None, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        """Create new SudokuDataset."""
        if data_url is None:
            assert subset is not None
            file_name = f"{subset.replace('sudoku_', '')}.parquet"
            self.data_url = f"{settings.data_url}/deep-protein-gen/sudoku_difficult/{file_name}"
        else:
            self.data_url = data_url
        super().__init__(root, transform, pre_transform, pre_filter)
        self.sudoku_graph = torch.from_numpy(gen_sudoku_graph_featured()).to_sparse()
        self.file = pq.ParquetFile(self.data_url)
        self.reset()

    def reset(self) -> None:
        self.prev_index = None
        self.row_group = None
        self.data_chunk = None
        self.data_chunk_idx = None

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        return self.file.metadata.num_rows

    def _read_row_group(self, row_group: int):
        df = self.file.read_row_group(row_group).to_pandas(integer_object_nulls=True)

        data_list = []
        for tup in df.itertuples():
            puzzle = str_to_tensor(tup.puzzle) - 1
            puzzle = torch.where(puzzle >= 0, puzzle, torch.tensor(9))
            solution = str_to_tensor(tup.solution) - 1
            data = Data(x=puzzle, y=solution)
            if self.pre_filter is not None:
                data = self.pre_filter(data)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def get(self, idx):
        if self.prev_index is None:
            assert idx == 0, idx
            self.row_group = 0
            self.data_chunk = self._read_row_group(self.row_group)
            self.data_chunk_idx = 0
        else:
            assert self.prev_index == idx - 1, (self.prev_index, idx)
            self.data_chunk_idx += 1
        self.prev_index = idx

        if self.data_chunk_idx >= len(self.data_chunk):
            self.row_group += 1
            self.data_chunk = self._read_row_group(self.row_group)
            self.data_chunk_idx = 0

        data = self.data_chunk[self.data_chunk_idx]
        data.edge_index = self.sudoku_graph.indices
        data.edge_attr = self.sudoku_graph.values
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


class SudokuDataset2(InMemoryDataset):
    def __init__(
        self,
        root,
        subset: Optional[str] = None,
        data_url: Optional[str] = None,
        make_local_copy: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        """Create new SudokuDataset."""
        self.data_url = (
            f"{settings.data_url}/deep-protein-gen/sudoku/sudoku_{subset}.csv.gz"
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
        df = df.rename(columns={"puzzle": "quizzes", "solution": "solutions"})

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
