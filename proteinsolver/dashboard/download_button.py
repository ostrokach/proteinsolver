from functools import partial

import ipywidgets as widgets
from IPython.display import HTML, display

from proteinsolver.dashboard.helper import save_sequences
from proteinsolver.dashboard.state import global_state


def create_download_button(output_folder):
    download_button = widgets.Button(
        description="Generate download link",
        tooltip="Generate download link",
        button_style="success",
        disabled=False,
        layout=widgets.Layout(width="auto"),
    )

    download_link_output = widgets.Output(layout=widgets.Layout(min_height="1.5rem"))

    download_button.on_click(
        partial(
            generate_download_link,
            download_link_output=download_link_output,
            output_folder=output_folder,
        )
    )

    return widgets.VBox([download_button, download_link_output])


def generate_download_link(download_button, download_link_output, output_folder):
    download_button.description = "Generating..."
    download_button.icon = "running"
    download_button.button_style = "info"  # 'success', 'info', 'warning', 'danger' or ''
    download_button.disabled = True

    output_file = save_sequences(global_state.generated_sequences, output_folder)

    download_link_output.clear_output(wait=True)
    with download_link_output:
        download_name = f"{output_file.stem[:8]}{output_file.suffix}"
        display(
            HTML(
                f'<a href="./voila/static/{output_file.name}" download{download_name}=>'
                f'<i class="fa fa-download"></i> Download sequences</a>'
            )
        )

    download_button.description = "Update download link"
    download_button.icon = ""  # check
    download_button.button_style = "success"
    download_button.disabled = False
