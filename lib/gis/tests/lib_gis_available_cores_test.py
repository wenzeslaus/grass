# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of G_available_cores()"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from grass.lib.gis import G_available_cores

needs_known_cores = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="the number of cores is unknown"
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
