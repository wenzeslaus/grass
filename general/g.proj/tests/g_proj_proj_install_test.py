# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: 2026 GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of g.proj -i, the report of the PROJ installation in use"""

import shutil
import sqlite3
from pathlib import Path

import pytest

import grass.script as gs
from grass.tools import ToolError, Tools

SHELL_KEYS = [
    "proj_runtime",
    "proj_db",
    "proj_db_rejected",
    "proj_db_layout",
    "proj_db_epsg",
    "proj_db_created_by_proj",
    "proj_searchpath",
]


def test_shell_keys_and_values(session):
    """All keys are present and describe a usable database"""
    info = gs.parse_command("g.proj", flags="i", format="shell", env=session.env)
    assert list(info.keys()) == SHELL_KEYS
    assert info["proj_runtime"].count(".") == 2
    assert info["proj_db"]
    assert not info["proj_db_rejected"]
    assert info["proj_searchpath"]
    if not Path(info["proj_db"]).is_file():
        pytest.skip("PROJ database is embedded in PROJ")
    assert Path(info["proj_db"]).name == "proj.db"
    assert info["proj_db_layout"].count(".") == 1
    assert info["proj_db_epsg"]
    assert info["proj_db_created_by_proj"]


def test_plain(session):
    """Plain output names the database in use"""
    info = gs.parse_command("g.proj", flags="i", format="shell", env=session.env)
    text = gs.read_command("g.proj", flags="i", env=session.env)
    assert info["proj_db"] in " ".join(text.split())


def test_unsupported_format(session):
    """Formats other than plain and shell are rejected"""
    with pytest.raises(ToolError):
        Tools(session=session).g_proj(flags="i", format="wkt")


def test_excludes_crs_input(session):
    """CRS input options cannot be combined with -i"""
    with pytest.raises(ToolError):
        Tools(session=session).g_proj(flags="i", epsg=4326)


def test_without_proj_db(env_without_proj_db, tools_without_proj_db):
    """The report works and says so when no proj.db can be found"""
    result = tools_without_proj_db.g_proj(flags="i", format="shell")
    assert result.returncode == 0, result.stderr
    info = result.keyval
    assert info["proj_db"] == ""
    assert info["proj_db_rejected"] == ""
    assert env_without_proj_db["PROJ_DATA"] in info["proj_searchpath"]

    result = tools_without_proj_db.g_proj(flags="i")
    assert result.returncode == 0, result.stderr
    assert "not found" in result.text


def test_with_incompatible_proj_db(
    env_without_proj_db, tools_without_proj_db, write_incompatible_proj_db
):
    """A proj.db which PROJ rejects is reported as such"""
    proj_db = Path(env_without_proj_db["PROJ_DATA"]) / "proj.db"
    write_incompatible_proj_db(proj_db)
    result = tools_without_proj_db.g_proj(flags="i", format="shell")
    assert result.returncode == 0, result.stderr
    info = result.keyval
    assert info["proj_db"] == ""
    assert info["proj_db_rejected"] == str(proj_db)

    result = tools_without_proj_db.g_proj(flags="i")
    assert result.returncode == 0, result.stderr
    assert "another PROJ installation" in " ".join(result.text.split())


def test_database_from_other_proj_version(session, env_without_proj_db):
    """A usable database created by another PROJ version is marked"""
    info = gs.parse_command("g.proj", flags="i", format="shell", env=session.env)
    if not Path(info["proj_db"]).is_file():
        pytest.skip("PROJ database is embedded in PROJ")
    env = env_without_proj_db
    proj_db = Path(env["PROJ_DATA"]) / "proj.db"
    shutil.copyfile(info["proj_db"], proj_db)
    with sqlite3.connect(proj_db) as connection:
        connection.execute(
            "UPDATE metadata SET value = '1.0.0' WHERE key = 'PROJ.VERSION'"
        )
    connection.close()

    other = gs.parse_command("g.proj", flags="i", format="shell", env=env)
    assert other["proj_db"] == str(proj_db)
    assert other["proj_db_created_by_proj"] == "1.0.0"
    text = " ".join(gs.read_command("g.proj", flags="i", env=env).split())
    assert "differs from the PROJ version in use" in text
