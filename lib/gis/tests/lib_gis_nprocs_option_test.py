# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of G_nprocs_from_option() and G_resolve_nprocs()"""

import subprocess
import sys

import pytest

from grass.lib.gis import G_available_cores
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


def test_white_space_around_answer(xy_session_for_module):
    """White space around a number, as in an NPROCS setting, is accepted"""
    returncode, output, stderr = nprocs_from_option(xy_session_for_module, " 2 \n")
    assert returncode == 0, stderr
    assert output.split()[0] == "2"


def test_tool_refuses_trailing_text(xy_session_for_module):
    """A tool refuses 2x although the parser lets it through"""
    tools = Tools(session=xy_session_for_module)
    with pytest.raises(ToolError) as error:
        tools.r_univar(map="doesnotexist", flags="g", nprocs="2x")
    assert "Invalid number of threads <2x> for nprocs=" in error.value.stderr


needs_known_cores = pytest.mark.skipif(
    G_available_cores() < 1, reason="the number of cores is unknown"
)


@pytest.fixture(scope="module")
def has_openmp(xy_session_for_module):
    """Whether G_set_omp_num_threads() sets up more than one thread"""
    _, output, _ = nprocs_from_option(xy_session_for_module, "2")
    return output == "2 2"


@needs_known_cores
@pytest.mark.parametrize("value", [None, "0", "abc", "-2", "3x", ""])
def test_cores_are_the_default(xy_session_for_module, value):
    """0 gives the cores without a positive OMP_NUM_THREADS, negative values count down

    Ten digits, which do not count as set either, are left out because the
    OpenMP runtime of LLVM aborts the process with them.
    """
    cores = G_available_cores()
    for answer, expected in [("0", cores), ("-1", max(1, cores - 1))]:
        returncode, output, stderr = nprocs_from_option(
            xy_session_for_module, answer, value
        )
        assert returncode == 0, stderr
        assert int(output.split()[0]) == expected
    _, output, _ = nprocs_from_option(xy_session_for_module, str(-cores - 5), value)
    assert int(output.split()[0]) == 1


@pytest.mark.parametrize(
    ("value", "answer", "expected"),
    [
        ("6", "0", 6),
        ("6", "-2", 4),
        ("6", "3", 3),
        ("6,abc", "0", 6),
        (" 6\n", "0", 6),
    ],
)
def test_omp_num_threads(xy_session_for_module, has_openmp, value, answer, expected):
    """The first value of a positive OMP_NUM_THREADS is the default

    Whatever follows the first value is ignored. A list of valid values,
    such as 6,2, is left out because it also turns on nested parallelism
    in the OpenMP runtime. Without OpenMP, G_set_omp_num_threads() returns 1.
    """
    returncode, output, stderr = nprocs_from_option(
        xy_session_for_module, answer, value
    )
    assert returncode == 0, stderr
    assert output == f"{expected} {expected if has_openmp else 1}"
