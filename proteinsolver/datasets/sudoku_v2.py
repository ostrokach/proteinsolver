import math
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset

from proteinsolver import settings
from proteinsolver.datasets import download_url
from proteinsolver.datasets.sudoku import str_to_tensor
from proteinsolver.utils import construct_solved_sudoku, gen_sudoku_graph, gen_sudoku_graph_featured


class SudokuDataset4(torch.utils.data.IterableDataset):
    def __init__(
        self, root, subset=None, data_url=None, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        """Create new SudokuDataset."""
        super().__init__()
        self.root = Path(root).expanduser().resolve().as_posix()
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if data_url is None:
            assert subset is not None
            file_name = f"{subset.replace('sudoku_', '')}.parquet"
            self.data_url = f"{settings.data_url}/deep-protein-gen/sudoku_difficult/{file_name}"
        else:
            self.data_url = data_url

        self.sudoku_graph = torch.from_numpy(gen_sudoku_graph_featured()).to_sparse(2)
        self.file = pq.ParquetFile(self.data_url)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_row_groups_per_worker = int(
                math.ceil(self.file.num_row_groups / worker_info.num_workers)
            )
            min_row_group_index = worker_info.id * num_row_groups_per_worker
            max_row_group_index = min(
                (worker_info.id + 1) * num_row_groups_per_worker, self.file.num_row_groups
            )
            row_group_indices = [
                i
                for i in range(0, self.file.num_row_groups)
                if min_row_group_index <= i < max_row_group_index
            ]
        else:
            row_group_indices = range(0, self.file.num_row_groups)

        for row_group in row_group_indices:
            data_list = self._read_row_group(row_group)
            for data in data_list:
                data.edge_index = self.sudoku_graph.indices()
                data.edge_attr = self.sudoku_graph.values()
                yield data

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
