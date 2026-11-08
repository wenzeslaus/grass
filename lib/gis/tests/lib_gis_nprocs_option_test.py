"""Tests of G_nprocs_from_option()"""

import subprocess
import sys

import pytest

from grass.lib.gis import G_available_cores

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
def session(xy_session_for_module):
    """Session in which G_gisinit() can run"""
    return xy_session_for_module


def nprocs(session, answer, omp_num_threads=None):
    """Return code, result of G_nprocs_from_option() and stderr for an answer"""
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


@pytest.mark.parametrize("answer", ["1", "3", "200"])
def test_positive(session, answer):
    """Positive numbers are used as given"""
    assert nprocs(session, answer)[:2] == (0, answer)


def test_default_is_physical_cores(session):
    """0 is the number of physical cores without OMP_NUM_THREADS"""
    assert nprocs(session, "0")[:2] == (0, str(G_available_cores()))


def test_negative_from_cores(session):
    """Negative numbers are that many fewer than the cores, at least 1"""
    cores = G_available_cores()
    assert nprocs(session, "-1")[:2] == (0, str(max(1, cores - 1)))
    assert nprocs(session, str(-cores - 5))[:2] == (0, "1")


@pytest.mark.parametrize(
    ("omp_num_threads", "expected"), [("6", 6), ("6,2", 6), ("0", None), ("x", None)]
)
def test_omp_num_threads(session, omp_num_threads, expected):
    """A positive OMP_NUM_THREADS replaces the cores as the default"""
    default = expected or G_available_cores()
    assert nprocs(session, "0", omp_num_threads)[:2] == (0, str(default))
    assert nprocs(session, "-2", omp_num_threads)[:2] == (0, str(max(1, default - 2)))
    assert nprocs(session, "3", omp_num_threads)[:2] == (0, "3")


@pytest.mark.parametrize("answer", ["", "two", "2x", "99999999999"])
def test_invalid(session, answer):
    """Answers which are not integers are errors naming the option"""
    returncode, unused, stderr = nprocs(session, answer)
    assert returncode != 0
    assert "nprocs=" in stderr


@pytest.mark.parametrize(
    "omp_num_threads", [None, "6", "6,2", " 6 ", "+6", "0", "x", "3x"]
)
def test_resolve_nprocs_same_as_python(monkeypatch, omp_num_threads):
    """G_resolve_nprocs() follows the same rules as grass.script"""
    import grass.script as gs
    from grass.lib.gis import G_resolve_nprocs

    if omp_num_threads is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", omp_num_threads)
    for value in [-100, -3, -1, 0, 1, 7]:
        assert G_resolve_nprocs(value) == gs.resolve_nprocs(value)
