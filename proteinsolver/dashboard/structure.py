import codecs
import io
from pathlib import Path

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
    data = proteinsolver.datasets.protein.row_to_data(pdata)
    data = proteinsolver.datasets.protein.transform_edge_attr(data)

    global_state.structure = domain
    global_state.data = data
    global_state.reference_sequence = list(proteinsolver.utils.array_to_seq(data.x))
    global_state.target_sequence = ["-"] * len(global_state.reference_sequence)


def update_displayed_structure(ngl_stage):
    if ngl_stage.n_components:
        ngl_stage.remove_component(ngl_stage.component_0)
    ngl_stage.add_component(PDB.structure_to_ngl(global_state.structure))


def create_load_structure_button(ngl_stage, target_selection_out, sequence_generation_out):
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

    uploader = FileUpload(accept=".pdb,.cif,.mmcif", multiple=False)
    uploader.observe(handle_upload, names="value")
    return uploader


def create_load_example_buttons(ngl_stage, target_selection_out, sequence_generation_out):
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

        button = Button(description=filename.stem)
        button.on_click(on_example_clicked)
        return button

    buttons = [create_activate_example_button(example) for example in examples]
    line = HBox(buttons, layout=Layout(flex_flow="row", align_items="center"))
    return line
