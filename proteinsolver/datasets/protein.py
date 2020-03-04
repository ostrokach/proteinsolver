import os.path as osp
import pickle
from pathlib import Path
from typing import Iterator, List, Mapping, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, InMemoryDataset

from proteinsolver import settings
from proteinsolver.datasets import download_url
from proteinsolver.utils import seq_to_tensor

_data_urls = {
    "train_0": (
        f"{settings.data_url}/deep-protein-gen/processed/training_data_0/"
        "part-00000-a260936c-8c1c-4b93-b9ab-57757dbf29b8-c000.snappy.parquet"
    ),
    "train_1": (
        f"{settings.data_url}/deep-protein-gen/processed/training_data_1/"
        "part-00000-7cab69dd-7eec-4823-8c4b-c355264bca9b-c000.snappy.parquet"
    ),
    "valid": (
        f"{settings.data_url}/deep-protein-gen/processed/validation_data/"
        "part-00000-4f535e50-cdf4-4275-b6b3-a3038f24a1a9-c000.snappy.parquet"
    ),
    "test": (
        f"{settings.data_url}/deep-protein-gen/processed/test_data/"
        "part-00000-ba92a066-6ee2-47dc-883c-fd2044ecaa00-c000.snappy.parquet"
    ),
}


class ProteinDataset(Dataset):
    subset: str
    data_url: str
    extra_columns: List[str]
    extra_column_renames: Mapping[str, str]
    _num_rows: int

    def __init__(
        self,
        root,
        subset: str,
        data_url=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        extra_columns=None,
        extra_column_renames=None,
    ) -> None:
        self.subset = subset
        self.data_url = data_url if data_url is not None else _data_urls[subset]
        self.extra_columns = extra_columns if extra_columns is not None else []
        self.extra_column_renames = extra_column_renames if extra_column_renames is not None else {}
        self._raw_file_names = [self.data_url.rsplit("/")[-1]]
        try:
            with Path(root).joinpath("info.pkl").open("rb") as fin:
                self._num_examples = pickle.load(fin)["num_examples"]
                self._processed_file_names = [f"data_{i:07d}.pt" for i in range(self._num_examples)]
        except FileNotFoundError:
            self._num_examples = None
            self._processed_file_names = ["dummy"]
        transform = T.Compose(
            [transform_edge_attr] + ([transform] if transform is not None else [])
        )
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def __len__(self):
        if self._num_examples is not None:
            return self._num_examples
        else:
            raise RuntimeError("Don't know length untill we process all data!")

    def download(self):
        download_url(self.data_url, self.raw_dir)

    def process(self):
        i = 0
        self._processed_file_names = []
        for tup in iter_parquet_file(
            self.raw_paths[0], self.extra_columns, self.extra_column_renames
        ):
            data = row_to_data(tup)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            filename = f"data_{i:07d}.pt"
            torch.save(data, osp.join(self.processed_dir, filename))
            self._processed_file_names.append(filename)
            i += 1
        with Path(self.root).joinpath("info.pkl").open("wb") as fin:
            pickle.dump({"num_examples": i}, fin, pickle.HIGHEST_PROTOCOL)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx:07d}.pt"))
        return data


class ProteinInMemoryDataset(InMemoryDataset):
    subset: str
    data_url: str
    extra_columns: List[str]
    extra_column_renames: Mapping[str, str]
    _num_rows: int

    def __init__(
        self,
        root,
        subset: str,
        data_url=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        extra_columns=None,
        extra_column_renames=None,
    ) -> None:
        self.subset = subset
        self.data_url = data_url if data_url is not None else _data_urls[subset]
        self.extra_columns = extra_columns if extra_columns is not None else []
        self.extra_column_renames = extra_column_renames if extra_column_renames is not None else {}
        self._raw_file_names = [self.data_url.rsplit("/")[-1]]
        self._processed_file_names = [f"protein_{subset}.pt"]
        transform = T.Compose(
            [transform_edge_attr] + ([transform] if transform is not None else [])
        )
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def download(self):
        download_url(self.data_url, self.raw_dir)

    def process(self):
        data_list = []
        for tup in iter_parquet_file(
            self.raw_paths[0], self.extra_columns, self.extra_column_renames
        ):
            data = row_to_data(tup)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            if data is not None:
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def normalize_cart_distances(cart_distances):
    return (cart_distances - 6) / 12


def normalize_seq_distances(seq_distances):
    return (seq_distances - 0) / 68.1319


def transform_edge_attr(data):
    cart_distances = data.edge_attr
    cart_distances = normalize_cart_distances(cart_distances)
    seq_distances = (data.edge_index[1] - data.edge_index[0]).to(torch.float).unsqueeze(1)
    seq_distances = normalize_seq_distances(seq_distances)
    data.edge_attr = torch.cat([cart_distances, seq_distances], dim=1)
    return data


def iter_parquet_file(
    filename: Union[str, Path], extra_columns: List[str], extra_column_renames: Mapping[str, str]
) -> Iterator:
    columns = (
        ["sequence", "residue_idx_1_corrected", "residue_idx_2_corrected", "distances"]
        if not extra_columns
        else extra_columns
    )

    column_renames = (
        {"residue_idx_1_corrected": "row_index", "residue_idx_2_corrected": "col_index"}
        if not extra_column_renames
        else extra_column_renames
    )

    parquet_file_obj = pq.ParquetFile(filename)
    for row_group_idx in range(parquet_file_obj.num_row_groups):
        df = parquet_file_obj.read_row_group(row_group_idx, columns=columns).to_pandas()
        df = df.rename(columns=column_renames)
        for tup in df.itertuples():
            yield tup


def row_to_data(tup, add_reversed_edges=True) -> Data:
    seq = torch.tensor(
        seq_to_tensor(tup.sequence.replace("-", "").encode("ascii")), dtype=torch.long
    )
    if (seq == 20).sum() > 0:
        return None

    row_index = _to_torch(tup.row_index).to(torch.long)
    col_index = _to_torch(tup.col_index).to(torch.long)
    edge_attr = _to_torch(tup.distances).to(torch.float).unsqueeze(dim=1)

    # Remove self loops
    mask = row_index == col_index
    if mask.any():
        row_index = row_index[~mask]
        col_index = col_index[~mask]
        edge_attr = edge_attr[~mask, :]

    if add_reversed_edges:
        edge_index = torch.stack(
            [torch.cat([row_index, col_index]), torch.cat([col_index, row_index])], dim=0
        )
        edge_attr = torch.cat([edge_attr, edge_attr])
    else:
        edge_index = torch.stack([row_index, col_index], dim=0)

    edge_index, edge_attr = remove_nans(edge_index, edge_attr)
    data = Data(x=seq, edge_index=edge_index, edge_attr=edge_attr)
    data = data.coalesce()

    assert not data.contains_self_loops()
    assert data.is_coalesced()
    assert data.is_undirected()

    for c in tup._fields:
        if c not in ["sequence", "row_index", "col_index", "distances"]:
            setattr(data, c, torch.tensor([getattr(tup, c)]))

    return data


def remove_nans(edge_index, edge_attr):
    na_mask = torch.isnan(edge_index).any(dim=0).squeeze()
    if na_mask.any():
        edge_index = edge_index[:, ~na_mask]
        edge_attr = edge_attr[~na_mask]
    return edge_index, edge_attr


def _to_torch(data_array):
    if isinstance(data_array, torch.Tensor):
        return data_array
    elif isinstance(data_array, np.ndarray):
        return torch.from_numpy(data_array)
    else:
        return torch.tensor(data_array)
