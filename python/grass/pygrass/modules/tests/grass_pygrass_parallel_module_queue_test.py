# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Test the worker count of ParallelModuleQueue"""

import pytest

import grass.script as gs
from grass.pygrass.modules import ParallelModuleQueue


@pytest.mark.parametrize("nprocs", [None, 0])
def test_worker_count_follows_nprocs(nprocs):
    """0 and None give the default of nprocs, also when set later"""
    queue = ParallelModuleQueue(nprocs=nprocs)
    assert queue.get_max_num_procs() == gs.resolve_nprocs(0)
    queue.set_max_num_procs(nprocs)
    assert queue.get_max_num_procs() == gs.resolve_nprocs(0)
