import pytest
from kmbio import PDB

from proteinsolver.utils import protein_structure


@pytest.mark.parametrize("structure_id", ["4dkl"])
def test_extract_seq_and_adj(structure_id):
    structure = PDB.load(f"rcsb://{structure_id}.cif.gz")
    print(structure)
    assert protein_structure
