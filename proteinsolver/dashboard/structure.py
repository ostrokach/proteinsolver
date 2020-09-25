import codecs
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from IPython.display import clear_output, display
from ipywidgets import Button, FileUpload, HBox, Layout
from kmbio import PDB
from kmtools import structure_tools

import proteinsolver
from proteinsolver.dashboard import (
    global_state,
    update_sequence_generation,
    update_target_selection,
)


def load_structure(structure: PDB.Structure):
    chain_id = next(next(structure.models).chains).id

    domain, result_df = proteinsolver.utils.get_interaction_dataset_wdistances(
        structure, 0, chain_id, r_cutoff=12, remove_hetatms=True
    )
    domain_sequence = structure_tools.get_chain_sequence(domain)
    assert max(result_df["residue_idx_1"].values) < len(domain_sequence)
    assert max(result_df["residue_idx_2"].values) < len(domain_sequence)

    pdata = proteinsolver.utils.ProteinData(
        domain_sequence,
        result_df["residue_idx_1"].values,
        result_df["residue_idx_2"].values,
        result_df["distance"].values,
    )
    tdata = proteinsolver.datasets.protein.row_to_data(pdata)
    data = proteinsolver.datasets.protein.transform_edge_attr(tdata.clone())

    global_state.structure = domain
    global_state.tdata = tdata
    global_state.data = data
    global_state.reference_sequence = list(proteinsolver.utils.array_to_seq(data.x))
    global_state.target_sequence = ["-"] * len(global_state.reference_sequence)


def load_distance_matrix(distance_matrix: str):
    # Parse distance matrix file
    num_residues = None
    results = []
    for line in distance_matrix.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("N:"):
            row = re.split(": *", line)
            num_residues = int(row[1])
        else:
            row = re.split(", *", line)
            residue_idx_1, residue_idx_2, distance = int(row[0]), int(row[1]), float(row[2])
            if residue_idx_1 == residue_idx_2:
                continue
            elif residue_idx_1 < residue_idx_2:
                results.append((residue_idx_1, residue_idx_2, distance))
            else:
                results.append((residue_idx_2, residue_idx_1, distance))

    # Remove duplicates
    results = list(set(results))

    if results:
        residue_idx_1_lst, residue_idx_2_lst, distance_lst = list(zip(*results))
    else:
        residue_idx_1_lst, residue_idx_2_lst, distance_lst = [], [], []

    if num_residues is None:
        num_residues = max(residue_idx_1_lst + residue_idx_2_lst)

    pdata = proteinsolver.utils.ProteinData(
        "G" * num_residues,
        np.array(residue_idx_1_lst),
        np.array(residue_idx_2_lst),
        np.array(distance_lst),
    )
    tdata = proteinsolver.datasets.protein.row_to_data(pdata)
    data = proteinsolver.datasets.protein.transform_edge_attr(tdata.clone())

    global_state.structure = None
    global_state.tdata = tdata
    global_state.data = data
    global_state.reference_sequence = list(proteinsolver.utils.array_to_seq(data.x))
    global_state.target_sequence = ["-"] * len(global_state.reference_sequence)


def update_displayed_structure(ngl_stage):
    if ngl_stage.n_components:
        ngl_stage.remove_component(ngl_stage.component_0)
    if global_state.structure is not None:
        ngl_stage.add_component(PDB.structure_to_ngl(global_state.structure))


def update_displayed_distance_matrix(distance_matrix_out):
    tdata = global_state.tdata
    adj = torch_geometric.utils.to_dense_adj(
        edge_index=tdata.edge_index, edge_attr=1 / tdata.edge_attr[:, 0]
    ).squeeze()

    fig = plt.figure(constrained_layout=False, figsize=(4 * 0.8, 3 * 0.8))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        top=0.98,
        right=0.85,
        bottom=0.15,
        left=0.1,
        hspace=0,
        wspace=0,
        width_ratios=[3, 0.1],  # 16
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    out = ax.imshow(adj, cmap="Greys")
    ax.set_ylabel("Amino acid position")
    ax.set_xlabel("Amino acid position")
    ax.tick_params("both")
    cb = fig.colorbar(out, cax=cax)
    cb.set_label("1 / distance (Å$^{-1}$)")

    with distance_matrix_out:
        clear_output()
        display(fig, display_id="distance-matrix")


def create_load_structure_button(
    ngl_stage, distance_matrix_out, target_selection_out, sequence_generation_out
):
    uploader = FileUpload(
        description="Load structure",
        accept=".pdb,.cif,.mmcif",
        multiple=False,
        layout=Layout(width="11rem"),
    )

    def handle_upload(change):
        # Keep only the last file (there must be a better way!)
        last_item = list(change["new"].values())[-1]

        filename = last_item["metadata"]["name"]
        structure_id = filename.split(".")[0]
        suffix = filename.split(".")[-1]

        data = codecs.decode(last_item["content"], encoding="utf-8")
        buf = io.StringIO()
        buf.write(data)
        buf.seek(0)
        parser = PDB.get_parser(suffix)
        structure = parser.get_structure(buf, structure_id=structure_id)

        # TODO: We may need to lock global_state at this point?
        load_structure(structure)

        update_target_selection(target_selection_out)
        update_sequence_generation(sequence_generation_out)
        update_displayed_structure(ngl_stage)
        update_displayed_distance_matrix(distance_matrix_out)

        uploader.value.clear()
        uploader._counter = 0

    uploader.observe(handle_upload, names="value")
    return uploader


def create_load_distance_matrix_button(
    ngl_stage, distance_matrix_out, target_selection_out, sequence_generation_out
):
    uploader = FileUpload(
        description="Load distance matrix",
        accept=".txt",
        multiple=False,
        layout=Layout(width="11rem"),
    )

    def handle_upload(change):
        # Keep only the last file (there must be a better way!)
        last_item = list(change["new"].values())[-1]

        data = codecs.decode(last_item["content"], encoding="utf-8")

        # TODO: We may need to lock global_state at this point?
        load_distance_matrix(data)

        update_target_selection(target_selection_out)
        update_sequence_generation(sequence_generation_out)
        update_displayed_structure(ngl_stage)
        update_displayed_distance_matrix(distance_matrix_out)

        uploader.value.clear()
        uploader._counter = 0

    uploader.observe(handle_upload, names="value")
    return uploader


def create_load_example_buttons(
    ngl_stage, distance_matrix_out, target_selection_out, sequence_generation_out
):
    examples_folder = (
        Path(proteinsolver.__path__[0]).resolve(strict=True).joinpath("data", "inputs")
    )
    examples = [
        examples_folder.joinpath(file)
        for file in ["1n5uA03.pdb", "4beuA02.pdb", "4unuA00.pdb", "4z8jA00.pdb"]
    ]

    def create_activate_example_button(filename):
        def on_example_clicked(change):
            structure = PDB.load(filename)
            # TODO: We may need to lock global_state at this point?
            load_structure(structure)
            update_target_selection(target_selection_out)
            update_sequence_generation(sequence_generation_out)
            update_displayed_structure(ngl_stage)
            update_displayed_distance_matrix(distance_matrix_out)

        button = Button(description=filename.stem, layout=Layout(width="8.25rem"))
        button.on_click(on_example_clicked)
        return button

    buttons = [create_activate_example_button(example) for example in examples]
    line = HBox(buttons, layout=Layout(flex_flow="row", align_items="center"))
    return line
