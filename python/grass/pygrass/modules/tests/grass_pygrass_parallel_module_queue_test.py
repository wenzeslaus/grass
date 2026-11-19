# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Test the worker count of ParallelModuleQueue"""

import grass.script as gs
from grass.pygrass.modules import ParallelModuleQueue


def test_worker_count_follows_nprocs(tmp_path):
    """0 gives the default and None gives NPROCS, also when set later"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project):
        gs.run_command("g.gisenv", set="NPROCS=3")
        for nprocs, expected in [(0, gs.resolve_nprocs(0)), (None, 3)]:
            queue = ParallelModuleQueue(nprocs=nprocs)
            assert queue.get_max_num_procs() == expected
            queue.set_max_num_procs(nprocs)
            assert queue.get_max_num_procs() == expected
