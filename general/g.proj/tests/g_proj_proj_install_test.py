"""Tests of g.proj -i, the report of the PROJ installation in use

Uses the session fixture from conftest.py.
"""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

import grass.script as gs

SHELL_KEYS = [
    "proj_runtime",
    "proj_db",
    "proj_db_rejected",
    "proj_db_layout",
    "proj_db_epsg",
    "proj_db_created_by_proj",
    "proj_searchpath",
]


def run_g_proj(args, env):
    """Run g.proj, return the return code, stdout and stderr"""
    process = gs.Popen(
        ["g.proj", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


@pytest.fixture
def env_without_proj_db(session, tmp_path):
    """Environment in which PROJ cannot find proj.db (see conftest.py)"""
    env = session.env.copy()
    proj_data = tmp_path / "proj_data"
    user_dir = tmp_path / "user_dir"
    proj_data.mkdir()
    user_dir.mkdir()
    # PROJ 9.0 uses only PROJ_LIB.
    env["PROJ_DATA"] = str(proj_data)
    env["PROJ_LIB"] = str(proj_data)
    env["PROJ_USER_WRITABLE_DIRECTORY"] = str(user_dir)
    info = gs.parse_command("g.proj", flags="i", format="shell", env=env)
    if info["proj_db"]:
        pytest.skip("PROJ database cannot be hidden (embedded in PROJ)")
    return env, proj_data


def test_shell_keys_and_values(session):
    """All keys are present and describe a usable database"""
    info = gs.parse_command("g.proj", flags="i", format="shell", env=session.env)
    assert list(info.keys()) == SHELL_KEYS
    assert info["proj_runtime"].count(".") == 2
    assert info["proj_db"]
    assert not info["proj_db_rejected"]
    if not Path(info["proj_db"]).is_file():
        pytest.skip("PROJ database is embedded in PROJ")
    assert Path(info["proj_db"]).name == "proj.db"
    assert info["proj_db_layout"].count(".") == 1
    assert info["proj_db_epsg"]
    assert info["proj_db_created_by_proj"]
    assert info["proj_searchpath"]


def test_plain(session):
    """Plain output names the database"""
    text = gs.read_command("g.proj", flags="i", env=session.env)
    assert "PROJ version:" in text
    assert "Database:" in text


def test_unsupported_format(session):
    """Formats other than plain and shell are rejected"""
    returncode, _, stderr = run_g_proj(["-i", "format=wkt"], env=session.env)
    assert returncode != 0
    assert "ERROR:" in stderr


def test_excludes_crs_input(session):
    """CRS input options cannot be combined with -i"""
    returncode, _, stderr = run_g_proj(["-i", "epsg=4326"], env=session.env)
    assert returncode != 0
    assert "ERROR:" in stderr


def test_without_proj_db(env_without_proj_db):
    """The report works and says so when no proj.db can be found"""
    env, proj_data = env_without_proj_db
    returncode, stdout, stderr = run_g_proj(["-i", "format=shell"], env=env)
    assert returncode == 0, stderr
    info = gs.parse_key_val(stdout)
    assert info["proj_db"] == ""
    assert info["proj_db_rejected"] == ""
    assert str(proj_data) in info["proj_searchpath"]

    returncode, stdout, stderr = run_g_proj(["-i"], env=env)
    assert returncode == 0, stderr
    assert "not found" in stdout


def test_with_incompatible_proj_db(env_without_proj_db):
    """A proj.db which PROJ rejects is reported as such"""
    env, proj_data = env_without_proj_db
    proj_db = proj_data / "proj.db"
    with sqlite3.connect(proj_db) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        connection.execute(
            "INSERT INTO metadata VALUES ('DATABASE.LAYOUT.VERSION.MAJOR', '999')"
        )
    connection.close()
    returncode, stdout, stderr = run_g_proj(["-i", "format=shell"], env=env)
    assert returncode == 0, stderr
    info = gs.parse_key_val(stdout)
    assert info["proj_db"] == ""
    assert info["proj_db_rejected"] == str(proj_db)

    returncode, stdout, stderr = run_g_proj(["-i"], env=env)
    assert returncode == 0, stderr
    assert "another PROJ installation" in " ".join(stdout.split())


def test_database_from_other_proj_version(session, env_without_proj_db):
    """A usable database created by another PROJ version is flagged"""
    info = gs.parse_command("g.proj", flags="i", format="shell", env=session.env)
    if not Path(info["proj_db"]).is_file():
        pytest.skip("PROJ database is embedded in PROJ")
    env, proj_data = env_without_proj_db
    proj_db = proj_data / "proj.db"
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
