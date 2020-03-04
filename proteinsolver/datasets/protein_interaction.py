"""

Example protein interaction RecordBatch:

.. code-block:: json

    {
        "pdb_id": "10gs",
        "pdb_idx": 25,
        "use_auth_id": False,
        "bioassembly_id": True,
        "model_id_1": 0,
        "chain_idx_1": 0,
        "chain_id_1": "A",
        "chain_idx_2": 1,
        "chain_id_2": "B",
        "num_interacting_residue_pairs": 82,
        "aa_sequence_1":
        "PYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGRTLGLYGKDQQ...",
        "aa_sequence_2":
        "PYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGRTLGLYGKDQQ...",
        "residue_idx": [...],  # <pyarrow.lib.UInt16Array object at 0x2abc283c5520>
        "dssp": ["C", "E", "E", ...],  # <pyarrow.lib.StringArray object at 0x2abc283c5360>
        "phi": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c53d0>
        "psi": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5440>
        "omega_prev": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c54b0>
        "omega_next": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5600>
        "ca_angles": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5670>
        "ca_dihedral_prev": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c56e0>
        "ca_dihedral_next": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5750>
        "chain_idx": [...],  # <pyarrow.lib.UInt16Array object at 0x2abc283c57c0>
        "residue_idx_1": [...],  # <pyarrow.lib.UInt16Array object at 0x2abc283c58a0>
        "residue_idx_2": [...],  # <pyarrow.lib.UInt16Array object at 0x2abc283c58a0>
        "distance": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5910>
        "distance_backbone": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5980>
        "distance_ca": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c59f0>
        "hbond": [...],  # <pyarrow.lib.BooleanArray object at 0x2abc283c5a60>
        "translation_1": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5ad0>
        "translation_2": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5b40>
        "rotation_1": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5bb0>
        "rotation_2": [...],  # <pyarrow.lib.FloatArray object at 0x2abc283c5c20>
    }
"""
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import pyarrow as pa
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset

from proteinsolver.datasets.protein import row_to_data, transform_edge_attr
from proteinsolver.utils.compression import decompress


class ProteinInteractionTuple(NamedTuple):
    sequence: str
    row_index: torch.Tensor
    col_index: torch.Tensor
    distances: torch.Tensor
    sequence_1_length: torch.Tensor

    @classmethod
    def from_rb_dict(cls, rb_dict: Mapping[str, Any]) -> "ProteinInteractionTuple":
        return cls(
            sequence=rb_dict["aa_sequence_1"] + rb_dict["aa_sequence_2"],
            row_index=torch.from_numpy(rb_dict["residue_idx_1"].astype(np.int64)),
            col_index=torch.from_numpy(rb_dict["residue_idx_2"].astype(np.int64)),
            distances=torch.from_numpy(rb_dict["distance"].astype(np.float32)),
            sequence_1_length=torch.tensor([len(rb_dict["aa_sequence_1"])]),
        )


class ProteinInteractionDataset(Dataset):
    def __init__(self, input_files, transform=None) -> None:
        if isinstance(input_files, (str, Path)):
            input_files = [input_files]
        readers = [pa.RecordBatchFileReader(input_file) for input_file in input_files]
        self.readers = [(reader, reader.num_record_batches) for reader in readers]
        transform = T.Compose(
            [transform_edge_attr] + ([transform] if transform is not None else [])
        )
        super().__init__(Path.cwd(), transform, None, None)

    def get(self, idx: int) -> Data:
        for reader, reader_size in self.readers:
            if idx - reader_size >= 0:
                idx -= reader_size
                continue
            rb = reader.get_batch(idx)
            break
        rb_dict = self.decode_record_batch_partial(rb)
        tup = ProteinInteractionTuple.from_rb_dict(rb_dict)
        data = row_to_data(tup, add_reversed_edges=False)
        return data

    def __len__(self) -> int:
        return sum(reader_size for _, reader_size in self.readers)

    def decode_record_batch_partial(self, rb) -> Mapping[str, Any]:
        data = {}
        for col_idx, col_name in [
            (10, "aa_sequence_1"),
            (11, "aa_sequence_2"),
            (12, "residue_idx"),
            (21, "chain_idx"),
            (22, "residue_idx_1"),
            (23, "residue_idx_2"),
            (24, "distance"),
        ]:
            value = rb.column(col_idx)
            assert len(value) == 1
            value = value[0].as_py()
            if isinstance(value, bytes):
                value = decompress(value).to_numpy()
            data[col_name] = value
        return data

    def decode_record_batch(self, rb) -> Mapping[str, Any]:
        data = {}
        for key, value in rb.to_pydict().items():
            assert len(value) == 1
            value = value[0].as_py()
            if isinstance(value, bytes):
                value = decompress(value).to_numpy()
            data[key] = value
        return data

    def _download(self):
        pass

    def _process(self):
        pass
