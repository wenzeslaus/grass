# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of G_available_cores() and of the threads G_set_omp_num_threads() sets"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from grass.lib.gis import G_available_cores

needs_known_cores = pytest.mark.skipif(
    G_available_cores() < 1, reason="the number of cores is unknown"
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
@pytest.mark.parametrize("cores", [1, 2])
def test_cores_in_affinity_mask_count_once(cores):
    """A process limited to one or two cores has that many cores

    The first core is one allowed CPU with its SMT siblings when the mask
    has them, the second one is another allowed CPU. G_available_cores()
    runs in a subprocess, so the affinity of the test process does not change.
    """
    allowed = os.sched_getaffinity(0)
    try:
        cpus = max((siblings(cpu) & allowed for cpu in allowed), key=len)
    except OSError:
        cpus = {min(allowed)}
    if cores == 2:
        if cpus == allowed:
            pytest.skip("needs two cores in the affinity mask")
        cpus.add(max(allowed - cpus))
    code = (
        "import os\n"
        f"os.sched_setaffinity(0, {cpus})\n"
        "from grass.lib.gis import G_available_cores\n"
        "print(G_available_cores())\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert int(output) == cores


def test_default_is_read_once():
    """A call with nprocs=-1 counts from the default, not from an earlier result

    Calls G_set_omp_num_threads() three times in one process through ctypes.
    """
    code = (
        "from ctypes import byref\n"
        "from grass.lib.gis import G_set_omp_num_threads, Option, String\n"
        "option = Option()\n"
        "option.key = String(b'nprocs')\n"
        "for answer in (b'4', b'-1', b'-1'):\n"
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
    assert threads == [4, 2, 2]
