import io
from pathlib import Path

import kmbio.PDB
from kmtools import structure_tools
from tkpod.plugins.modeller import Modeller

from .adjacency_matrix import get_interaction_dataset


def get_modeller_scores(row):
    structure_data = row.structure_text
    fh = io.StringIO()
    fh.write(structure_data)
    fh.seek(0)

    parser = kmbio.PDB.PDBParser()
    structure = parser.get_structure(fh, bioassembly_id=False)
    chain = structure[0][row.chain_id]
    sequence = structure_tools.get_chain_sequence(chain)

    target = structure_tools.DomainTarget(0, row.chain_id, sequence, 1, len(sequence), sequence)
    modeller_data = Modeller.build(structure, bioassembly_id=False, use_strict_alignment=True)
    structure_bm, modeller_results = Modeller.create_model([target], modeller_data)

    return structure_bm, sequence, modeller_results


def get_pdb_interactions(row, pdb_ffindex_path: Path):
    structure_id, _, chain_id = row.template_id.partition("_")
    structure_url = f"ff://{pdb_ffindex_path}?{structure_id.lower()}.cif.gz"

    with kmbio.PDB.open_url(structure_url) as fin:
        structure = kmbio.PDB.MMCIFParser().get_structure(fin)
        fin.seek(0)
        structure_dict = kmbio.PDB.mmcif2dict(fin)

    chain_sequence = structure_tools.get_chain_sequence(structure[0][chain_id])

    if isinstance(structure_dict["_entity_poly.pdbx_strand_id"], str):
        construct_sequence = structure_dict["_entity_poly.pdbx_seq_one_letter_code_can"]
    else:
        seq_idx = [
            i
            for i, v in enumerate(structure_dict["_entity_poly.pdbx_strand_id"])
            if chain_id in v.split(",")
        ][0]
        construct_sequence = structure_dict["_entity_poly.pdbx_seq_one_letter_code_can"][seq_idx]

    try:
        offset = construct_sequence.index(chain_sequence[:10])
    except ValueError:
        offset = 0

    target_1 = structure_tools.DomainTarget(
        0,
        chain_id,
        row.template_ali,
        row.template_start - offset,
        row.template_end - offset,
        row.query_ali,
    )

    modeller_data = Modeller.build(
        structure_url, use_auth_id=True, bioassembly_id=False, use_strict_alignment=False
    )

    # ---

    structure_fixed = kmbio.PDB.load(
        modeller_data.structure_file, bioassembly_id=modeller_data.bioassembly_id
    )
    structure_fixed_cut, alignment = structure_tools.prepare_for_modeling(
        structure_fixed, [target_1], strict=modeller_data.use_strict_alignment
    )

    interactions_core, interactions_core_aggbychain = get_interaction_dataset(structure_fixed_cut)
    return interactions_core, interactions_core_aggbychain


def get_homology_model_interactions(row):
    structure_data = row.structure_text
    fh = io.StringIO()
    fh.write(structure_data)
    fh.seek(0)

    parser = kmbio.PDB.PDBParser()
    structure = parser.get_structure(fh)
    structure.id = row.unique_id
    interactions_core, interactions_core_aggbychain = get_interaction_dataset(structure)
    return interactions_core, interactions_core_aggbychain
