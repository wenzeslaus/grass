"""Test grass.tools.StandaloneTools class with NumPy arrays"""

import numpy as np
import pytest

from grass.tools import StandaloneTools


def test_numpy_multiple_inputs_one_output():
    """Check that a NumPy array works for multiple inputs"""
    tools = StandaloneTools()
    result = tools.r_mapcalc_simple(
        expression="A + B", a=np.full((2, 3), 2), b=np.full((2, 3), 5), output=np.array
    )
    # Size of output
    assert result.shape == (2, 3)
    assert np.all(result == np.full((2, 3), 7))
    # Computational region
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 3
    result = tools.r_mapcalc_simple(expression="int(10)", output=np.array)
    assert result.shape == (2, 3)
    assert np.all(result == np.full((2, 3), 10))


def test_numpy_manual_region_mismatch():
    """Check that a NumPy array works for multiple inputs"""
    tools = StandaloneTools()
    result = tools.r_mapcalc_simple(
        expression="A + B", a=np.full((2, 3), 2), b=np.full((2, 3), 5), output=np.array
    )
    assert result.shape == (2, 3)
    region = tools.g_region(rows=2, cols=2)
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 2
    with pytest.raises(ValueError, match=r"broadcast.*shape.*\(2.*3\).*\(2.*2\)"):
        tools.r_mapcalc_simple(
            expression="A + B",
            a=np.full((2, 3), 2),
            b=np.full((2, 3), 5),
            output=np.array,
        )


def test_numpy_workflow_region_mismatch():
    """Check that a NumPy array works for multiple inputs"""
    tools = StandaloneTools()
    result = tools.r_mapcalc_simple(
        expression="A + B", a=np.full((2, 3), 2), b=np.full((2, 3), 5), output=np.array
    )
    assert result.shape == (2, 3)
    with pytest.raises(ValueError, match=r"broadcast.*shape.*\(4.*5\).*\(2.*3\)"):
        tools.r_mapcalc_simple(
            expression="A + B",
            a=np.full((4, 5), 2),
            b=np.full((4, 5), 5),
            output=np.array,
        )


def test_numpy_inputs_region_mismatch():
    """Check that a NumPy array works for multiple inputs"""
    tools = StandaloneTools()
    with pytest.raises(ValueError, match=r"broadcast.*shape.*\(4.*5\).*\(2.*3\)"):
        tools.r_mapcalc_simple(
            expression="A + B",
            a=np.full((2, 3), 2),
            b=np.full((4, 5), 5),
            output=np.array,
        )
