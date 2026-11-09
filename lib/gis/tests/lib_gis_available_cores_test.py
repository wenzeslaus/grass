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

needs_known_cores = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="the number of cores is unknown on this platform",
)


@needs_known_cores
def test_between_one_and_logical_cpus():
    """There is at least one core and not more cores than logical CPUs"""
    assert 1 <= G_available_cores() <= os.cpu_count()


def siblings(cpu):
    """Logical CPUs on the same core as cpu (SMT siblings), from sysfs"""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    cpus = set()
    for part in path.read_text(encoding="utf-8").strip().split(","):
        first, unused, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity (Linux)"
)
def test_core_in_affinity_mask_counts_once():
    """A process limited to one core has one core

    The core is one allowed CPU with its SMT siblings when the mask has
    them. G_available_cores() runs in a subprocess, so the affinity of the
    test process does not change.
    """
    allowed = os.sched_getaffinity(0)
    try:
        cpus = max((siblings(cpu) & allowed for cpu in allowed), key=len)
    except OSError:
        cpus = {min(allowed)}
    code = (
        "import os\n"
        f"os.sched_setaffinity(0, {cpus})\n"
        "from grass.lib.gis import G_available_cores\n"
        "print(G_available_cores())\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert int(output) == 1


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


@needs_known_cores
def test_default_is_available_cores(univar_session):
    """Without OMP_NUM_THREADS, nprocs=0 gives one thread per core"""
    assert thread_count(univar_session, nprocs=0) == G_available_cores()


def test_default_is_read_once():
    """A second call with nprocs=-1 counts from the default, not from the first result

    Calls G_set_omp_num_threads() three times in one process through ctypes.
    """
    code = (
        "from ctypes import byref\n"
        "from grass.lib.gis import G_gisinit, G_set_omp_num_threads, Option, String\n"
        "G_gisinit(b'test')\n"
        "option = Option()\n"
        "option.key = String(b'nprocs')\n"
        "for answer in (b'3', b'-1', b'-1'):\n"
        "    option.answer = String(answer)\n"
        "    print(G_set_omp_num_threads(byref(option)))\n"
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
    if threads[0] == 1:
        pytest.skip("GRASS is built without OpenMP")
    assert threads == [3, 2, 2]
