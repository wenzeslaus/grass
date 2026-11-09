# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Test the worker count of ParallelModuleQueue"""

import pytest

from grass.pygrass.modules import ParallelModuleQueue
from grass.script import utils as gutils


@pytest.mark.parametrize(
    ("nprocs", "expected"), [(1, 1), (3, 3), (0, 8), (None, 8), (-2, 6)]
)
def test_nprocs(monkeypatch, nprocs, expected):
    """0 and None give the default number of cores, negative values reserve"""
    monkeypatch.setattr(gutils, "available_cores", lambda: 8)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert ParallelModuleQueue(nprocs=nprocs)._num_procs == expected
