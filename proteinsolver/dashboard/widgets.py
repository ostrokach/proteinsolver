import threading
import time

import ipywidgets as widgets
from IPython.display import HTML, display

from proteinsolver.dashboard import generate_random_sequence, load_structure


def populate_generated_sequences():
    global generated_sequences

    sequences = []
    for i in range(20_000):
        sequence = {
            "id": i + 1,
            "name": f"gen-{i:05d}",
            "proba": 1.0,
            "seq": generate_random_sequence(162),
        }
        sequences.append(sequence)


def create_progress_bar():
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        step=1,
        bar_style="",  # 'success', 'info', 'warning', 'danger' or ''
        orientation="horizontal",
        layout=widgets.Layout(width="auto", height="10px"),
    )
