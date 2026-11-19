# SPDX-FileCopyrightText: 2025-2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
"""Test grass.tools.StandaloneTools class with NumPy arrays"""

import numpy as np
import pytest

from grass.tools import StandaloneTools


def test_numpy_multiple_inputs_one_output(tmp_path):
    """Check that arrays work as inputs and output and the first input sets the region"""
    tools = StandaloneTools(work_dir=tmp_path)
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


def test_numpy_manual_region_mismatch(tmp_path):
    """Check that inputs fail when their shape differs from a region set with g.region"""
    tools = StandaloneTools(work_dir=tmp_path)
    result = tools.r_mapcalc_simple(
        expression="A + B", a=np.full((2, 3), 2), b=np.full((2, 3), 5), output=np.array
    )
    assert result.shape == (2, 3)
    tools.g_region(rows=2, cols=2)
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


def test_numpy_workflow_region_mismatch(tmp_path):
    """Check that inputs fail when their shape differs from inputs of an earlier call"""
    tools = StandaloneTools(work_dir=tmp_path)
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


def test_numpy_inputs_region_mismatch(tmp_path):
    """Check that an input fails when its shape differs from the first input"""
    tools = StandaloneTools(work_dir=tmp_path)
    with pytest.raises(ValueError, match=r"broadcast.*shape.*\(4.*5\).*\(2.*3\)"):
        tools.r_mapcalc_simple(
            expression="A + B",
            a=np.full((2, 3), 2),
            b=np.full((4, 5), 5),
            output=np.array,
        )
