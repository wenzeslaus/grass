# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Test the worker count of ParallelModuleQueue"""

import pytest

import grass.script as gs
from grass.pygrass.modules import ParallelModuleQueue


@pytest.mark.parametrize("nprocs", [None, 0, -1, 2])
def test_worker_count_follows_nprocs(nprocs):
    """0, None, and negative values are resolved like nprocs"""
    queue = ParallelModuleQueue(nprocs=nprocs)
    assert queue.get_max_num_procs() == gs.resolve_nprocs(nprocs or 0)
