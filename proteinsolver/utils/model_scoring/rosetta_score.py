import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def get_rosetta_scores(row):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path.joinpath(row.unique_id + ".pdb")
        with input_file.open("wt") as fout:
            fout.write(row.structure_text)
        relax_df = run_rosetta_relax(input_file, temp_path)
        input_files = [input_file] + list(input_file.parent.glob(input_file.stem + "_*.pdb"))
        assert len(input_files) > 1
        score_jd2_df = run_rosetta_score(input_files, temp_path)
    relax_best_row = relax_df[(relax_df["total_score"] == relax_df["total_score"].min())]
    score_best_row = score_jd2_df[
        (score_jd2_df["total_score"] == score_jd2_df["total_score"].min())
    ]
    rosetta_results = {
        **{"rosetta_relax_" + k: v for k, v in dict(relax_best_row.mean()).items()},
        **{"rosetta_score_" + k: v for k, v in dict(score_best_row.mean()).items()},
    }
    return rosetta_results


def run_rosetta_relax(input_file, temp_path):
    output_file = temp_path.joinpath("relax.sc")
    if output_file.is_file():
        output_file.unlink()

    system_command = [
        f"{os.environ['ROSETTA_BIN']}/relax.static.linuxgccrelease",
        f"-in:file:s '{input_file}'",
        f"-out:file:scorefile '{output_file}'",
        "-overwrite",
        "-ignore_unrecognized_res",
        "-nstruct 3",
    ]
    # Fix backbone:
    system_command += ["--relax:constrain_relax_to_start_coords", "--relax:ramp_constraints false"]
    # Fix side-chains:
    # system_command += ["--relax:coord_constrain_sidechains"]

    proc = subprocess.run(
        shlex.split(" ".join(system_command)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=temp_path,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise Exception("Rosetta relax crashed!")

    relax_df = pd.read_csv(temp_path.joinpath("relax.sc"), sep=" +", engine="python", skiprows=1)
    del relax_df["SCORE:"]
    return relax_df


def run_rosetta_score(input_files, temp_path):
    output_file = temp_path.joinpath("score_jd2.sc")
    if output_file.is_file():
        output_file.unlink()

    input_files_string = " ".join(["'{}'".format(f) for f in input_files])
    system_command = [
        f"{os.environ['ROSETTA_BIN']}/score_jd2.static.linuxgccrelease",
        f"-in:file:s {input_files_string}",
        f"-out:file:scorefile '{output_file}'",
        "-overwrite",
    ]

    proc = subprocess.run(
        shlex.split(" ".join(system_command)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=temp_path,
        check=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise Exception("Rosetta score crashed!")

    score_jd2_df = pd.read_csv(
        temp_path.joinpath("score_jd2.sc"), sep=" +", engine="python", skiprows=1
    )
    del score_jd2_df["SCORE:"]
    return score_jd2_df
