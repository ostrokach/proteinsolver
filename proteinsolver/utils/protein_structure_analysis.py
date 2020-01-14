import itertools

import mdtraj
import numpy as np
import pandas as pd
import torch
from kmtools import structure_tools

# === Features describing individual residues ===


def calculate_sasa(traj: mdtraj.Trajectory) -> pd.DataFrame:
    """Return absolute and relative SASA value for each residue.

    Args:
        traj: Trajectory for the protein of interest.

    Returns:
        Pandas dataframe with columns: [residue_index, sasa, relative_sasa].
    """
    results = []
    for residue, sasa in zip(traj.topology.residues, mdtraj.shrake_rupley(traj, mode="residue")[0]):
        results.append((sasa, sasa * 100 / structure_tools.constants.STANDARD_SASA[residue.name],))
    assert len(results) == traj.topology.n_residues
    return np.array(results)


def calculate_phi(traj: mdtraj.Trajectory) -> pd.DataFrame:
    """Calculate phi angles for all residues.

    Args:
        traj: Trajectory for the protein of interest.

    Returns:
        Pandas dataframe with columns: [residue_index, phi].
    """
    results = [np.nan]
    indices, angles = mdtraj.compute_phi(traj)
    for index, angle in zip(indices, angles[0]):
        assert (
            traj.topology.atom(index[0]).residue.index + 1
            == traj.topology.atom(index[1]).residue.index
        )
        assert (
            traj.topology.atom(index[1]).residue
            == traj.topology.atom(index[2]).residue
            == traj.topology.atom(index[3]).residue
        )
        results.append(angle)
    assert len(results) == traj.topology.n_residues
    return np.array(results)


def calculate_psi(traj: mdtraj.Trajectory) -> pd.DataFrame:
    """Calculate psi angles for all residues.

    Args:
        traj: Trajectory for the protein of interest.

    Returns:
        Pandas dataframe with columns: [residue_index, psi].
    """
    results = []
    indices, angles = mdtraj.compute_psi(traj)
    for index, angle in zip(indices, angles[0]):
        assert (
            traj.topology.atom(index[0]).residue
            == traj.topology.atom(index[1]).residue
            == traj.topology.atom(index[2]).residue
        )
        assert (
            traj.topology.atom(index[2]).residue.index + 1
            == traj.topology.atom(index[3]).residue.index
        )
        results.append(angle)
    results.append(np.nan)
    assert len(results) == traj.topology.n_residues
    return np.array(results)


def calculate_omega(traj: mdtraj.Trajectory) -> pd.DataFrame:
    """Calculate preceding and succeeding omega angles for all residues.

    Args:
        traj: Trajectory for the protein of interest.

    Returns:
        Pandas dataframe with columns: [residue_index, omega_prev, omega_next].
    """
    results = [(np.nan, np.nan)]
    indices, angles = mdtraj.compute_omega(traj)
    for index, angle in zip(indices, angles[0]):
        #     print([traj.topology.atom(i).residue for i in index])
        assert (
            traj.topology.atom(index[1]).residue.index + 1
            == traj.topology.atom(index[2]).residue.index
        )
        assert traj.topology.atom(index[0]).residue == traj.topology.atom(index[1]).residue
        assert traj.topology.atom(index[2]).residue == traj.topology.atom(index[3]).residue
        results[-1] = results[-1][:-1] + (angle,)
        results.append((angle, np.nan))
    results[-1] = results[-1][:-1] + (np.nan,)
    return np.array(results)


def _get_ca_atom_indices(traj):
    residue_indices, ca_atom_indices = mdtraj.geometry.dihedral._atom_sequence(
        traj.topology, ["CA"]
    )
    assert (residue_indices == np.arange(residue_indices.size)).all()
    return ca_atom_indices


def calculate_backbone_angles(traj: mdtraj.Trajectory) -> pd.DataFrame:
    ca_atom_indices = _get_ca_atom_indices(traj)
    angle_indices = np.hstack([ca_atom_indices[:-2], ca_atom_indices[1:-1], ca_atom_indices[2:]])
    angles = mdtraj.compute_angles(traj, angle_indices)[0]
    angles = np.pad(angles, 1, constant_values=np.nan)
    return angles.astype(np.double)


def calculate_backbone_dihedrals(traj: mdtraj.Trajectory):
    ca_atom_indices = _get_ca_atom_indices(traj)
    angle_indices = np.hstack(
        [ca_atom_indices[:-3], ca_atom_indices[1:-2], ca_atom_indices[2:-1], ca_atom_indices[3:]]
    )
    dihedrals = mdtraj.compute_dihedrals(traj, angle_indices)[0]
    dihedrals = np.pad(dihedrals, 2, constant_values=np.nan)
    dihedrals = np.c_[dihedrals[:-1], dihedrals[1:]]
    return dihedrals.astype(np.double)


# === Features describing interactions between residues ===


def get_internal_coords(df: pd.DataFrame) -> np.ndarray:
    """

    Args:
        df: Pandas dataframe with columns:
            ["residue_idx", "atom_name", "atom_x", "atom_y", "atom_z"].
        normed: If `True`, return an orthonormal basis (each 3D vector has an l2 norm of 1).

    Returns:
        Numpy array with three vectors describing the orientation of each residue.
    """
    df_sele = df[df["atom_name"].isin({"N", "CA", "C"})]
    assert (
        df_sele["atom_name"]
        == np.array(list(itertools.islice(itertools.cycle(["N", "CA", "C"]), df_sele.shape[0])))
    ).all()
    ar = df_sele[["atom_x", "atom_y", "atom_z"]].values
    ar_diff = ar[1:, :] - ar[:-1, :]
    n_ca = ar_diff[0::3]
    ca_c = ar_diff[1::3]
    u = ca_c - n_ca
    v = np.cross(ca_c, n_ca)
    w = np.cross(v, u)
    assert len(df["residue_idx"].drop_duplicates()) == u.shape[0] == v.shape[0] == w.shape[0]
    internal_coords = np.stack([u, v, w], axis=1)
    internal_coords = _normalize_coords(internal_coords)
    return internal_coords


def _normalize_coords(coords):
    coords_unfolded = coords.reshape(coords.shape[0] * 3, 3)
    coords_unfolded_normed = coords_unfolded / np.sqrt(
        (coords_unfolded ** 2).sum(axis=1, keepdims=True)
    )
    assert np.allclose(np.sqrt((coords_unfolded_normed ** 2).sum(axis=1)), 1)
    return coords_unfolded_normed.reshape(*coords.shape)


def get_translations(df: pd.DataFrame) -> torch.Tensor:
    """Return translations for every pair of residues in `df`.

    Args:
        df: DataFrame returned by `kmbio.PDB.to_dataframe()`.

    Returns:
        PyTorch tensor of shape `N x N x 3`.
    """
    df_sele = df[(df["atom_name"].isin({"CA"}))]
    ar = df_sele[["atom_x", "atom_y", "atom_z"]].values
    translations = ar[None, :, :] - ar[:, None, :]
    assert len(df["residue_idx"].drop_duplicates()) == translations.shape[0]
    assert (
        (translations[np.arange(translations.shape[0]), np.arange(translations.shape[1])] == 0)
        .all()
        .item()
    )
    return translations


def map_translations_to_internal_coords(
    translations: torch.Tensor, internal_coords: torch.Tensor
) -> torch.Tensor:
    translations_internal = np.matmul(translations, internal_coords)
    assert (
        (
            translations_internal[
                np.arange(translations.shape[0]), np.arange(translations.shape[1])
            ]
            == 0
        )
        .all()
        .item()
    )
    return translations_internal


def get_rotations(internal_coords):
    """Return rotation matrices for every pair of residues in `internal_coords`.

    Returns:
        PyTorch tensor of shape `N x N x 3 x 3`.
    """
    # The PyTorch method:
    # internal_coords = torch.from_numpy(internal_coords)
    # rotations = torch.matmul(internal_coords.transpose(-1, -2).unsqueeze(1), internal_coords)
    rotations = np.matmul(internal_coords.transpose(0, 2, 1)[:, None, :, :], internal_coords)
    # The dot product of orthonormal matrices should be an identity matrix.
    diags = rotations[torch.arange(rotations.shape[0]), torch.arange(rotations.shape[1])]
    eyes = np.eye(3, dtype=np.double)[None, :, :].repeat(diags.shape[0], axis=0)
    assert np.allclose(diags, eyes)
    return rotations


def calculate_hydrogen_bonds(traj: mdtraj.Trajectory) -> pd.DataFrame:
    """Return a list of residues connected by hydrogen bonds.

    Args:
        traj: Trajectory for the protein of interest.

    Returns:
        Pandas dataframe with columns: [residue_index_1, residue_index_2].
    """
    d_h_a = mdtraj.wernet_nilsson(traj)[0]
    if not d_h_a.size:
        return pd.DataFrame([], columns=["residue_index_1", "residue_index_2"])
    residue_pairs = [
        (traj.topology.atom(d_i).residue.index, traj.topology.atom(a_i).residue.index)
        for d_i, _, a_i in d_h_a
    ]
    residue_pairs = sorted({(r_1, r_2) if r_1 <= r_2 else (r_2, r_1) for r_1, r_2 in residue_pairs})
    return pd.DataFrame(residue_pairs, columns=["residue_index_1", "residue_index_2"])
