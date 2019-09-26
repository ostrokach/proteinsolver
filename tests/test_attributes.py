import pytest

import proteinsolver


@pytest.mark.parametrize("attribute", ["__version__"])
def test_attribute(attribute):
    assert getattr(proteinsolver, attribute)


def test_main():
    import proteinsolver

    assert proteinsolver
