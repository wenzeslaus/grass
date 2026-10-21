"""Tests of G_available_cores()"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from grass.lib.gis import G_available_cores


def available_logical_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


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
    cpu = min(os.sched_getaffinity(0))
    try:
        cpus = siblings(cpu) & os.sched_getaffinity(0)
    except OSError:
        pytest.skip("no CPU topology in sysfs")
    if len(cpus) < 2:
        pytest.skip("no SMT siblings available")
    assert cores_with_affinity(cpus) == 1
