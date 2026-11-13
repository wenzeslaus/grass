# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Test the worker count of ParallelModuleQueue"""

import pytest

from grass.pygrass.modules import ParallelModuleQueue
from grass.script import utils as gutils


@pytest.mark.parametrize("nprocs", [0, None])
def test_default_worker_count(monkeypatch, nprocs):
    """0 and None give one worker per core"""
    monkeypatch.setattr(gutils, "available_cores", lambda: 8)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert ParallelModuleQueue(nprocs=nprocs).get_max_num_procs() == 8
