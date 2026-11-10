# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of G_nprocs_from_option() and G_resolve_nprocs()"""

import subprocess
import sys

import pytest

from grass.lib.gis import G_available_cores, G_resolve_nprocs
from grass.tools import ToolError, Tools

# Prints G_nprocs_from_option() and G_set_omp_num_threads() for the answer
# in argv[1].
SCRIPT = """
import sys

from grass.lib.ctypes_preamble import String
from grass.lib.gis import (
    G_OPT_M_NPROCS,
    G_define_standard_option,
    G_gisinit,
    G_nprocs_from_option,
    G_set_omp_num_threads,
)

G_gisinit("lib_gis_nprocs_option_test")
option = G_define_standard_option(G_OPT_M_NPROCS)
option.contents.answer = String(sys.argv[1].encode())
print(G_nprocs_from_option(option), G_set_omp_num_threads(option))
"""


def nprocs_from_option(session, answer, omp_num_threads=None):
    """Exit code, output of SCRIPT, and stderr for an answer

    OpenMP reads OMP_NUM_THREADS when it is loaded, so the variable is
    set (None: unset) for a new process.
    """
    env = session.env.copy()
    env.pop("OMP_NUM_THREADS", None)
    if omp_num_threads is not None:
        env["OMP_NUM_THREADS"] = omp_num_threads
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, answer],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr


def test_out_of_range(xy_session_for_module):
    """An answer beyond the int range is an error naming the option"""
    returncode, _, stderr = nprocs_from_option(xy_session_for_module, "99999999999")
    assert returncode != 0
    assert "Invalid number of threads <" in stderr
    assert "nprocs=" in stderr


def test_tool_refuses_trailing_text(xy_session_for_module):
    """A tool refuses 2x although the parser lets it through"""
    tools = Tools(session=xy_session_for_module)
    with pytest.raises(ToolError) as error:
        tools.r_univar(map="doesnotexist", flags="g", nprocs="2x")
    assert "Invalid number of threads <" in error.value.stderr
    assert "nprocs=" in error.value.stderr


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="the number of cores is unknown on this platform",
)
def test_cores_are_the_default(monkeypatch):
    """Without OMP_NUM_THREADS, 0 gives the cores and negative values count down to 1"""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    cores = G_available_cores()
    assert G_resolve_nprocs(0) == cores
    assert G_resolve_nprocs(-1) == max(1, cores - 1)
    assert G_resolve_nprocs(-cores - 5) == 1


@pytest.mark.parametrize(
    ("answer", "expected"), [("0", "6 6"), ("-2", "4 4"), ("3", "3 3")]
)
def test_omp_num_threads(xy_session_for_module, answer, expected):
    """With OMP_NUM_THREADS=6, the default is 6, and OpenMP gets the result"""
    returncode, output, stderr = nprocs_from_option(xy_session_for_module, answer, "6")
    if output.endswith(" 1"):
        pytest.skip("GRASS is built without OpenMP")
    assert (returncode, output) == (0, expected), stderr
