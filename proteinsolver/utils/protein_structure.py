from typing import NamedTuple

import torch
from kmtools import structure_tools


class ProteinData(NamedTuple):
    sequence: str
    row_index: torch.LongTensor
    col_index: torch.LongTensor
    distances: torch.FloatTensor


def extract_seq_and_adj(structure, chain_id, remove_hetatms=False):
    domain, result_df = get_interaction_dataset_wdistances(
        structure, 0, chain_id, r_cutoff=12, remove_hetatms=remove_hetatms
    )
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


def get_interaction_dataset_wdistances(
    structure, model_id, chain_id, r_cutoff=12, remove_hetatms=False
):
    chain = structure[0][chain_id]
    num_residues = len(list(chain.residues))
    dd = structure_tools.DomainDef(model_id, chain_id, 1, num_residues)
    domain = structure_tools.extract_domain(structure, [dd], remove_hetatms=remove_hetatms)
    distances_core = structure_tools.get_distances(
        domain.to_dataframe(), r_cutoff, groupby="residue"
    )
    assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
    return domain, distances_core
