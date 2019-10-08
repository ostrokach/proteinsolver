from typing import Optional

import pyarrow.parquet as pq
import torch_geometric.transforms as T
from proteinsolver.datasets import download_url
from proteinsolver.datasets.protein import iter_parquet_file, row_to_data, transform_edge_attr
from torch_geometric.data import Dataset


class ProteinDataset2(Dataset):
    def __init__(
        self,
        root,
        subset: Optional[str] = None,
        data_url: Optional[str] = None,
        make_local_copy: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        """Create new SudokuDataset."""
        if data_url is None:
            assert subset is not None
            i = int(subset.split("_")[-1])
            self.data_url = (
                f"https://storage.googleapis.com/deep-protein-gen/training_data_rs{i}.parquet"
            )
        else:
            self.data_url = data_url
        self.make_local_copy = make_local_copy
        self._raw_file_names = [self.data_url.rsplit("/")[-1]]
        pre_transform = T.Compose(
            [transform_edge_attr] + ([pre_transform] if pre_transform is not None else [])
        )
        super().__init__(root, transform, pre_transform, pre_filter)
        #
        self.file = pq.ParquetFile(self.input_file_path)
        self.data_iterator = iter_parquet_file(self.input_file_path, [], {})
        self.reset_parameters()

    def reset_parameters(self):
        self.prev_index = None
        self.row_group = None
        self.data_chunk_idx = None

    @property
    def raw_file_names(self):
        return self._raw_file_names

    def download(self):
        download_url(self.data_url, self.raw_dir)

    def _download(self):
        if self.make_local_copy:
            return super()._download()

    @property
    def processed_file_names(self):
        return []

    def _process(self):
        pass

    @property
    def input_file_path(self):
        if self.make_local_copy:
            return self.raw_paths[0]
        else:
            return self.data_url

    def __len__(self):
        return self.file.metadata.num_rows

    def get(self, idx):
        if self.prev_index is None:
            assert idx == 0, idx
            self.row_group = 0
            self.data_chunk_idx = 0
        else:
            assert self.prev_index == idx - 1
            self.data_chunk_idx += 1
        self.prev_index = idx

        while True:
            tup = next(self.data_iterator)
            data = row_to_data(tup)
            if data is None:
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            return data
