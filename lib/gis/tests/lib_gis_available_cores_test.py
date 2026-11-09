# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of G_available_cores() and of the threads G_set_omp_num_threads() sets"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from grass.lib.gis import G_available_cores
from grass.tools import Tools


def available_logical_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def test_between_one_and_logical_cpus():
    """There is at least one core and not more cores than logical CPUs"""
    assert 1 <= G_available_cores() <= available_logical_cpus()


def cores_with_affinity(cpus):
    """G_available_cores() in a process limited to the given logical CPUs"""
    code = (
        "import os\n"
        f"os.sched_setaffinity(0, {set(cpus)})\n"
        "from grass.lib.gis import G_available_cores\n"
        "print(G_available_cores())\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return int(output)


def siblings(cpu):
    """Logical CPUs on the same core as cpu (SMT siblings), from sysfs"""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    cpus = set()
    for part in path.read_text(encoding="utf-8").strip().split(","):
        first, unused, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


needs_affinity = pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity (Linux)"
)


@needs_affinity
def test_follows_affinity_mask():
    """With the process limited to one logical CPU, there is one core"""
    assert cores_with_affinity([min(os.sched_getaffinity(0))]) == 1


@needs_affinity
def test_counts_smt_siblings_once():
    """The logical CPUs of one core count as one core"""
    allowed = os.sched_getaffinity(0)
    try:
        groups = [siblings(cpu) & allowed for cpu in sorted(allowed)]
    except OSError:
        pytest.skip("no CPU topology in sysfs")
    cpus = next((group for group in groups if len(group) > 1), None)
    if not cpus:
        pytest.skip("no SMT siblings available")
    assert cores_with_affinity(cpus) == 1


def thread_count(session, nprocs, omp_num_threads=None):
    """Threads r.univar sets up with nprocs and OMP_NUM_THREADS (None: unset)

    Returns None when the message about threads is missing, which is the
    case without OpenMP.
    """
    env = session.env.copy()
    env.pop("OMP_NUM_THREADS", None)
    if omp_num_threads is not None:
        env["OMP_NUM_THREADS"] = omp_num_threads
    tools = Tools(env=env, verbose=True)
    stderr = tools.r_univar(map="one", flags="g", nprocs=nprocs).stderr
    if "One thread is set up" in stderr:
        return 1
    match = re.search(r"(\d+) threads are set up", stderr)
    return int(match.group(1)) if match else None


@pytest.fixture(scope="module")
def univar_session(xy_session_for_module):
    """Session with a 1x1 raster, skipping the tests without OpenMP"""
    session = xy_session_for_module
    tools = Tools(session=session)
    tools.g_region(rows=1, cols=1)
    tools.r_mapcalc(expression="one = 1")
    if thread_count(session, nprocs=2) is None:
        pytest.skip("GRASS is built without OpenMP")
    return session


@pytest.mark.parametrize(("nprocs", "expected"), [(0, 3), (-1, 2)])
def test_omp_num_threads_is_the_default(univar_session, nprocs, expected):
    """With OMP_NUM_THREADS=3, nprocs=0 gives 3 threads and nprocs=-1 gives 2"""
    assert thread_count(univar_session, nprocs=nprocs, omp_num_threads="3") == expected


def test_default_is_available_cores(univar_session):
    """Without OMP_NUM_THREADS, nprocs=0 gives one thread per core"""
    assert thread_count(univar_session, nprocs=0) == G_available_cores()


def test_default_is_read_once():
    """A second call with nprocs=-1 counts from the default, not from the first result

    Calls G_set_omp_num_threads() three times in one process through ctypes,
    with G_resolve_nprocs(-1) before and after.
    """
    code = (
        "from ctypes import byref\n"
        "from grass.lib.gis import (\n"
        "    G_gisinit, G_resolve_nprocs, G_set_omp_num_threads, Option, String\n"
        ")\n"
        "G_gisinit(b'test')\n"
        "print(G_resolve_nprocs(-1))\n"
        "option = Option()\n"
        "option.key = String(b'nprocs')\n"
        "for answer in (b'3', b'-1', b'-1'):\n"
        "    option.answer = String(answer)\n"
        "    print(G_set_omp_num_threads(byref(option)))\n"
        "print(G_resolve_nprocs(-1))\n"
    )
    env = os.environ | {"OMP_NUM_THREADS": "3"}
    output = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    threads = [int(line) for line in output.split()]
    if threads[1] == 1:
        pytest.skip("GRASS is built without OpenMP")
    assert threads == [2, 3, 2, 2, 2]
