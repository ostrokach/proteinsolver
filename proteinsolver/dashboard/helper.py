import gzip
import uuid
from pathlib import Path
from typing import List

import numpy as np

import proteinsolver
from proteinsolver.dashboard.msa_view import MSASeq


def generate_random_sequence(length=80, seed=None):
    amino_acids = np.array(proteinsolver.utils.AMINO_ACIDS)
    if seed is None:
        choice = np.random.choice
    else:
        choice = np.random.RandomState(seed).choice
    return "".join(choice(amino_acids, length))


def save_sequences(sequences: List[MSASeq], output_folder: Path) -> Path:
    sequences_fasta = sequences_to_fasta(sequences)
    sequences_fasta_gz = gzip.compress(sequences_fasta.encode("utf-8"))

    output_file = output_folder.joinpath(f"{uuid.uuid4()}.fasta.gz")
    with output_file.open("wb") as fout:
        fout.write(sequences_fasta_gz)

    return output_file


def sequences_to_fasta(sequences: List[MSASeq], line_width=80) -> str:
    sequence_string = ""
    for sequence in sequences:
        sequence_string += f">{sequence.id}|{sequence.name}|{sequence.proba}|{sequence.logproba}\n"
        for start in range(0, len(sequence.seq), line_width):
            sequence_string += sequence.seq[start : start + line_width] + "\n"
    return sequence_string
