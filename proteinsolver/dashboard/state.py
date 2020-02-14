from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch_geometric
from kmbio import PDB


@dataclass
class GlobalState:
    # Network initialization
    net_class: Optional[Callable]
    state_file: Optional[Union[str, Path]]
    net_kwargs: Optional[Dict[str, Any]]
    # Extract data from structure
    reference_sequence: List[str]
    target_sequence: List[str]
    data: Optional[torch_geometric.data.Data]
    structure: Optional[PDB.Structure]
    # Generate new sequences
    proteinsolver_thread: Any  # ProteinSolverThread
    generated_sequences: List[Dict]
    output_folder: Optional[Path]

    __slots__ = (
        "net_class",
        "state_file",
        "net_kwargs",
        "reference_sequence",
        "target_sequence",
        "data",
        "structure",
        "proteinsolver_process",
        "proteinsolver_thread",
        "generated_sequences",
        "output_folder",
    )


global_state = GlobalState(None, None, {}, [], [], None, None, None, [], None)
