from typing import Optional

import pyarrow.parquet as pq
import torch_geometric.transforms as T
from torch_geometric.data import Dataset

from proteinsolver import settings
from proteinsolver.datasets.protein import iter_parquet_file, row_to_data, transform_edge_attr


class ProteinDataset2(Dataset):
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
        if data_url is None:
            assert subset is not None
            file_name = f"training_data_rs{int(subset.split('_')[-1])}.parquet"
            self.data_url = f"{settings.data_url}/deep-protein-gen/{file_name}"
        else:
            self.data_url = data_url
        self._raw_file_names = [self.data_url.rsplit("/")[-1]]
        transform = T.Compose(
            [transform_edge_attr] + ([transform] if transform is not None else [])
        )
        super().__init__(root, transform, pre_transform, pre_filter)
        self.file = pq.ParquetFile(self.data_url)
        self.reset()

    def reset(self):
        self.data_iterator = iter_parquet_file(self.data_url, [], {})
        self.prev_index = None

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        # Warning: This over-estimates the number of data points because some rows are malformed
        return self.file.metadata.num_rows

    def get(self, idx):
        if self.prev_index is None:
            assert idx == 0, idx
        else:
            assert self.prev_index == idx - 1
        self.prev_index = idx

        while True:
            tup = next(self.data_iterator)
            data = row_to_data(tup)
            if data is None:
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            return data
