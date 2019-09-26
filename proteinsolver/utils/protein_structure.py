from typing import NamedTuple

import torch
from kmbio import PDB
from kmtools import structure_tools


class ProteinData(NamedTuple):
    sequence: str
    row_index: torch.LongTensor
    col_index: torch.LongTensor
    distances: torch.FloatTensor


def get_interaction_dataset_wdistances(structure_file, model_id, chain_id, r_cutoff=12):
    structure = PDB.load(structure_file)
    chain = structure[0][chain_id]
    num_residues = len(list(chain.residues))
    dd = structure_tools.DomainDef(model_id, chain_id, 1, num_residues)
    domain = structure_tools.extract_domain(structure, [dd])
    distances_core = structure_tools.get_distances(domain, r_cutoff, 0, groupby="residue")
    assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
    return domain, distances_core


def extract_seq_and_adj(structure_file, chain_id):
    domain, result_df = get_interaction_dataset_wdistances(structure_file, 0, chain_id, r_cutoff=12)
    domain_sequence = structure_tools.get_chain_sequence(domain)
    assert max(result_df["residue_idx_1"].values) < len(domain_sequence)
    assert max(result_df["residue_idx_2"].values) < len(domain_sequence)
    data = ProteinData(
        domain_sequence,
        result_df["residue_idx_1"].values,
        result_df["residue_idx_2"].values,
        result_df["distance"].values,
    )
    return data
