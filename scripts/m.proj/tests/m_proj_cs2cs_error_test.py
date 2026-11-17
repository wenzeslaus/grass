# SPDX-FileCopyrightText: 2026 GRASS Development Team
# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-License-Identifier: GPL-2.0-or-later

"""Test that m.proj fails when cs2cs fails"""

import os

import pytest

import grass.script as gs
from grass.tools import ToolError, Tools


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Active session in a WGS 84 project"""
    project = tmp_path_factory.mktemp("m_proj") / "wgs84"
    gs.create_project(project, epsg="4326")
    with gs.setup.init(project, env=os.environ.copy()) as session:
        yield session


def test_cs2cs_failure_is_error(session):
    """A PROJ string which cs2cs cannot use makes the tool fail"""
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="cs2cs exit code"):
        tools.m_proj(
            coordinates=(1, 2),
            proj_in="+proj=unknown",
            proj_out="+proj=longlat +datum=WGS84",
        )
