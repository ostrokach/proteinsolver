import ipywidgets as widgets
from IPython.display import HTML, display
from ipywidgets import Layout

import proteinsolver
from proteinsolver.dashboard.state import global_state


def update_target_selection(target_selection_out):
    html_string = (
        '<p class="myheading">'
        "2. (Optional) Specify target amino acids at specific positions "
        "(or enter '-' to leave the position open for design)"
        "</p>"
    )
    target_sequence_selection_widget = create_target_selection_widget()
    target_selection_out.clear_output(wait=True)
    with target_selection_out:
        display(HTML(html_string))
        display(target_sequence_selection_widget)


def create_target_selection_widget():
    def update_target_sequence(change):
        residue_idx = int(change["owner"].description.split(" ")[0])
        global_state.target_sequence[residue_idx] = change["new"]
        target_sequence_ta.value = "".join(global_state.target_sequence)

    reference_sequence_ta = widgets.Textarea(
        value="".join(global_state.reference_sequence),
        placeholder="AAAAA...",
        description="<em>Reference</em><br>sequence:",
        disabled=True,
        layout=widgets.Layout(width="auto"),
    )
    _ = reference_sequence_ta.add_class("mysequence")

    target_sequence_ta = widgets.Textarea(
        value="".join(global_state.target_sequence),
        placeholder="AAAAA...",
        description="<em>Target</em><br>sequence:",
        disabled=True,
        layout=widgets.Layout(width="auto"),
    )
    _ = target_sequence_ta.add_class("mysequence")

    target_sequence_dropdowns = []
    for i, (aa_ref, aa_target) in enumerate(
        zip(global_state.reference_sequence, global_state.target_sequence)
    ):
        dropdown = widgets.Dropdown(
            options=["-"] + proteinsolver.utils.AMINO_ACIDS,
            value=aa_target,
            description=f"{i} ({aa_ref})",
            #                 style={},
            layout=widgets.Layout(width="120px"),
            style={"font_family": "monospace", "font_weight": "bold"},
        )
        dropdown.observe(update_target_sequence, names="value")
        _ = dropdown.add_class("mytext")
        target_sequence_dropdowns.append(dropdown)

    target_sequence_dropdowns_wg = widgets.HBox(
        target_sequence_dropdowns, layout=widgets.Layout(width="100%", flex_flow="row wrap")
    )

    accordion = widgets.Accordion(
        children=[target_sequence_dropdowns_wg], layout=Layout(margin="2px 0px 0px 90px")
    )
    accordion.set_title(0, "Target residue picker")
    accordion.selected_index = None

    target_sequence_selection_wg = widgets.VBox(
        [widgets.VBox([reference_sequence_ta, target_sequence_ta]), accordion]
    )
    return target_sequence_selection_wg
