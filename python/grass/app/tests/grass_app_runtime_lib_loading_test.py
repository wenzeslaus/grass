# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of making the GRASS C libraries work without the setup done for a session

This file is meant to be executed also standalone (pytest with this file as
the only argument), including against an installation moved away from the
prefix it was configured with: the tests provide any needed environment
themselves instead of relying on the environment of the test process.
"""

import os
import sys
from pathlib import Path

import pytest

import grass.script as gs
from grass.app.runtime import RuntimePaths
from grass.script.setup import get_install_path
from grass.tools import Tools

IMPORT_CODE = """
    import json

    import grass.lib.gis
    import grass.lib.raster

    print(json.dumps({"imported": True}))
"""


def test_grass_lib_importable_without_session(
    run_code_in_environment, clean_subprocess_environment
):
    """Check the import with neither a session nor the search path variable"""
    assert run_code_in_environment(IMPORT_CODE, env=clean_subprocess_environment)[
        "imported"
    ]


def test_grass_lib_importable_with_manual_search_path(
    run_code_in_environment, clean_subprocess_environment
):
    """Check that the search path variable set at process start is enough"""
    lib_path = os.fspath(Path(get_install_path()) / "lib")
    env = clean_subprocess_environment
    variables = [RuntimePaths().ld_library_path_var]
    if sys.platform == "darwin":
        # The variable GRASS knows on macOS (LD_RUN_PATH) is read by the
        # ctypes loader but ignored by dyld, which reads DYLD_LIBRARY_PATH.
        variables.append("DYLD_LIBRARY_PATH")
    for variable in variables:
        if env.get(variable):
            env[variable] = lib_path + os.pathsep + env[variable]
        else:
            env[variable] = lib_path
    assert run_code_in_environment(IMPORT_CODE, env=env)["imported"]


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="On Windows, tools find the libraries through PATH",
)
def test_tool_runs_without_search_path_variable(tmp_path):
    """Check that a tool finds the GRASS libraries without the search path variable

    The tool runs in a session from which the search path variable set by
    the session is removed. The relative search path recorded in the tool
    is what the test checks only for an installation moved away from its
    prefix, otherwise the absolute one may be used.
    """
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        for variable in (RuntimePaths().ld_library_path_var, "DYLD_LIBRARY_PATH"):
            session.env.pop(variable, None)
        Tools(session=session).g_message(message="Libraries found")
