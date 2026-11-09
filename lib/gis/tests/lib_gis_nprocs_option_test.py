# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of G_nprocs_from_option() and G_resolve_nprocs()"""

import subprocess
import sys

import pytest

from grass.lib.gis import G_available_cores, G_resolve_nprocs
from grass.tools import ToolError, Tools

# Parses the answer of an nprocs option given on the command line.
SCRIPT = """
import sys

from grass.lib.ctypes_preamble import String
from grass.lib.gis import (
    G_OPT_M_NPROCS,
    G_define_standard_option,
    G_gisinit,
    G_nprocs_from_option,
)

G_gisinit("lib_gis_nprocs_option_test")
option = G_define_standard_option(G_OPT_M_NPROCS)
option.contents.answer = String(sys.argv[1].encode())
print(G_nprocs_from_option(option))
"""


@pytest.fixture
def no_omp_num_threads(monkeypatch):
    """Unset OMP_NUM_THREADS for the in-process calls"""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)


def nprocs_from_option(session, answer, omp_num_threads=None):
    """Return code, result of G_nprocs_from_option() and stderr for an answer

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


def test_option_answer(xy_session_for_module):
    """The answer of the option is parsed and resolved"""
    assert nprocs_from_option(xy_session_for_module, "3")[:2] == (0, "3")


@pytest.mark.parametrize("answer", ["", "two", "2x", "99999999999"])
def test_invalid(xy_session_for_module, answer):
    """Answers which are not integers are errors naming the option"""
    returncode, _, stderr = nprocs_from_option(xy_session_for_module, answer)
    assert returncode != 0
    assert "nprocs=" in stderr


def test_tool_refuses_trailing_text(xy_session_for_module):
    """A tool refuses 2x although the parser lets it through"""
    tools = Tools(session=xy_session_for_module)
    tools.g_region(rows=1, cols=1)
    tools.r_mapcalc(expression="one = 1")
    with pytest.raises(ToolError) as error:
        tools.r_univar(map="one", flags="g", nprocs="2x")
    assert "nprocs=" in error.value.stderr


@pytest.mark.usefixtures("no_omp_num_threads")
@pytest.mark.parametrize("value", [1, 3, 200])
def test_positive(value):
    """Positive numbers are used as given"""
    assert G_resolve_nprocs(value) == value


@pytest.mark.usefixtures("no_omp_num_threads")
def test_default_is_physical_cores():
    """0 is the number of physical cores without OMP_NUM_THREADS"""
    assert G_resolve_nprocs(0) == G_available_cores()


@pytest.mark.usefixtures("no_omp_num_threads")
def test_negative_from_cores():
    """Negative numbers are that many fewer than the cores, at least 1"""
    cores = G_available_cores()
    assert G_resolve_nprocs(-1) == max(1, cores - 1)
    assert G_resolve_nprocs(-cores - 5) == 1


@pytest.mark.parametrize(
    ("omp_num_threads", "answer", "expected"),
    [("6", "0", "6"), ("6,2", "0", "6"), ("6", "-2", "4"), ("6", "3", "3")],
)
def test_omp_num_threads(xy_session_for_module, omp_num_threads, answer, expected):
    """OMP_NUM_THREADS replaces the cores as the default"""
    result = nprocs_from_option(xy_session_for_module, answer, omp_num_threads)
    assert result[:2] == (0, expected)
