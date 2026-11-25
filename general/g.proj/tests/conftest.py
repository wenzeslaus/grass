# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: 2026 GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Fixtures for g.proj tests"""

import os
import sqlite3

import pytest

import grass.script as gs
from grass.tools import Tools


def _write_incompatible_proj_db(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("DATABASE.LAYOUT.VERSION.MAJOR", "999"),
                ("DATABASE.LAYOUT.VERSION.MINOR", "0"),
            ],
        )
    connection.close()


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Set up a GRASS session for the tests."""
    tmp_path = tmp_path_factory.mktemp("grass_session")
    project = "test_project"
    gs.create_project(tmp_path, project, epsg="3358")
    with gs.setup.init(tmp_path / project, env=os.environ.copy()) as session:
        yield session


@pytest.fixture
def env_without_proj_db(session, tmp_path):
    """Environment in which PROJ cannot find proj.db

    PROJ looks into the user-writable directory first and, when PROJ_DATA
    is set, only into the directories listed there, so pointing both to
    empty directories hides every proj.db on the machine. PROJ_LIB is set
    to the same directory for PROJ 9.0 which uses only PROJ_LIB.
    """
    env = session.env.copy()
    proj_data = tmp_path / "empty_proj_data"
    proj_data.mkdir()
    user_dir = tmp_path / "empty_user_dir"
    user_dir.mkdir()
    env["PROJ_DATA"] = str(proj_data)
    env["PROJ_LIB"] = str(proj_data)
    env["PROJ_USER_WRITABLE_DIRECTORY"] = str(user_dir)
    tools = Tools(env=env, errors="ignore", consistent_return_value=True)
    if tools.g_proj(flags="w", epsg=3358).returncode == 0:
        pytest.skip("PROJ database cannot be hidden (embedded in PROJ)")
    return env


@pytest.fixture
def tools_without_proj_db(env_without_proj_db):
    """Tools which run without proj.db and do not raise on errors"""
    return Tools(env=env_without_proj_db, errors="ignore", consistent_return_value=True)


@pytest.fixture
def write_incompatible_proj_db():
    """Function which writes a proj.db which PROJ rejects to the given path

    The database has a layout version of a future PROJ version.
    """
    return _write_incompatible_proj_db
