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
# in argv[1] (empty: an option without an answer).
SCRIPT = """
import sys
from ctypes import byref

from grass.lib.gis import G_nprocs_from_option, G_set_omp_num_threads, Option, String

option = Option()
option.key = String(b"nprocs")
option.answer = String(sys.argv[1].encode() or None)
print(G_nprocs_from_option(byref(option)), G_set_omp_num_threads(byref(option)))
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


@pytest.mark.parametrize("answer", ["99999999999", ""])
def test_not_an_int(xy_session_for_module, answer):
    """An answer beyond the int range or missing is an error naming the option"""
    returncode, _, stderr = nprocs_from_option(xy_session_for_module, answer)
    assert returncode != 0
    assert f"Invalid number of threads <{answer}> for nprocs=" in stderr


def test_tool_refuses_trailing_text(xy_session_for_module):
    """A tool refuses 2x although the parser lets it through"""
    tools = Tools(session=xy_session_for_module)
    with pytest.raises(ToolError) as error:
        tools.r_univar(map="doesnotexist", flags="g", nprocs="2x")
    assert "Invalid number of threads <2x> for nprocs=" in error.value.stderr


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


@pytest.mark.parametrize(("answer", "expected"), [("0", 6), ("-2", 4), ("3", 3)])
def test_omp_num_threads(xy_session_for_module, answer, expected):
    """With OMP_NUM_THREADS=6, the default is 6 for both functions

    Without OpenMP, G_set_omp_num_threads() returns 1.
    """
    returncode, output, stderr = nprocs_from_option(xy_session_for_module, answer, "6")
    assert returncode == 0, stderr
    _, openmp, _ = nprocs_from_option(xy_session_for_module, "2", "6")
    threads = expected if openmp == "2 2" else 1
    assert output == f"{expected} {threads}"


@pytest.mark.skipif(G_available_cores() < 1, reason="number of cores is unknown")
@pytest.mark.parametrize("value", ["0", "abc", "9999999999"])
def test_omp_num_threads_not_a_positive_number(xy_session_for_module, value):
    """With a value which does not count as set, 0 gives the cores"""
    returncode, output, stderr = nprocs_from_option(xy_session_for_module, "0", value)
    assert returncode == 0, stderr
    assert int(output.split()[0]) == G_available_cores()
