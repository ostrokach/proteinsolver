from typing import NamedTuple

import torch
from kmtools import structure_tools


class ProteinData(NamedTuple):
    sequence: str
    row_index: torch.LongTensor
    col_index: torch.LongTensor
    distances: torch.FloatTensor


def extract_seq_and_adj(structure, chain_id):
    domain, result_df = get_interaction_dataset_wdistances(structure, 0, chain_id, r_cutoff=12)
    domain_sequence = structure_tools.get_chain_sequence(domain)
    assert max(result_df["residue_idx_1"].values) < len(domain_sequence)
    assert max(result_df["residue_idx_2"].values) < len(domain_sequence)
    data = ProteinData(
        domain_sequence,
        result_df["residue_idx_1"].values,
        result_df["residue_idx_2"].values,
        result_df["distance"].values,
        # result_df["distance_backbone"].values,
        # result_df["orientation_1"].values,
        # result_df["orientation_2"].values,
        # result_df["orientation_3"].values,
    )
    return data


def get_interaction_dataset_wdistances(structure, model_id, chain_id, r_cutoff=12):
    chain = structure[0][chain_id]
    num_residues = len(list(chain.residues))
    dd = structure_tools.DomainDef(model_id, chain_id, 1, num_residues)
    domain = structure_tools.extract_domain(structure, [dd])
    distances_core = structure_tools.get_distances(domain, r_cutoff, 0, groupby="residue")
    assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
    return domain, distances_core
