"""Tests of g.proj behavior when PROJ cannot use its database"""

import sqlite3
import subprocess

import pytest

import grass.script as gs


def run_g_proj(args, env):
    """Run g.proj, return the return code and stderr with normalized spaces

    GRASS wraps long messages, so whitespace is collapsed to single spaces
    to make the text searchable.
    """
    process = gs.Popen(
        ["g.proj", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Paths with non-ASCII characters may not be in the expected encoding.
        errors="replace",
    )
    _, stderr = process.communicate()
    return process.returncode, " ".join(stderr.split())


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
    returncode, _ = run_g_proj(["-w", "epsg=3358"], env=env)
    if returncode == 0:
        pytest.skip("PROJ database cannot be hidden (embedded in PROJ)")
    return env, proj_data, user_dir


def write_incompatible_proj_db(path):
    """Write a proj.db which PROJ rejects, as if from a future PROJ version"""
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


def test_epsg_works_with_proj_db(session):
    """Sanity check that the EPSG code resolves in a normal environment"""
    output = gs.read_command("g.proj", flags="w", epsg="3358", env=session.env)
    assert "North Carolina" in output


def test_epsg_fails_with_hint_without_proj_db(env_without_proj_db):
    """Without proj.db, g.proj fails with a message naming the searched paths

    The return code must be 1 (a GRASS fatal error), not a crash.
    """
    env, proj_data, _ = env_without_proj_db
    returncode, stderr = run_g_proj(["-w", "epsg=3358"], env=env)
    assert returncode == 1, stderr
    assert "ERROR:" in stderr
    assert "proj.db" in stderr
    assert "PROJ_DATA" in stderr or "PROJ_LIB" in stderr
    assert str(proj_data) in stderr, "The message should name the searched path"


def test_epsg_fails_with_hint_for_incompatible_proj_db(env_without_proj_db):
    """A proj.db which PROJ rejects is reported as found but unusable

    This is the case of PROJ_DATA pointing to a database from another
    PROJ installation, simulated by a database with a future layout.
    """
    env, proj_data, _ = env_without_proj_db
    proj_db = proj_data / "proj.db"
    write_incompatible_proj_db(proj_db)
    returncode, stderr = run_g_proj(["-w", "epsg=3358"], env=env)
    assert returncode == 1, stderr
    assert "ERROR:" in stderr
    assert "another PROJ installation" in stderr
    assert str(proj_db) in stderr, "The message should name the rejected file"


def test_epsg_fails_with_hint_for_proj_lib_only(env_without_proj_db):
    """When only PROJ_LIB is set, the message names PROJ_LIB and its value"""
    env, proj_data, _ = env_without_proj_db
    del env["PROJ_DATA"]
    returncode, stderr = run_g_proj(["-w", "epsg=3358"], env=env)
    assert returncode == 1, stderr
    assert f"PROJ_LIB is set to <{proj_data}>" in stderr


def test_epsg_fails_with_hint_for_incompatible_proj_db_in_user_dir(
    env_without_proj_db,
):
    """A rejected proj.db in the PROJ user directory is to be removed"""
    env, _, user_dir = env_without_proj_db
    proj_db = user_dir / "proj.db"
    write_incompatible_proj_db(proj_db)
    returncode, stderr = run_g_proj(["-w", "epsg=3358"], env=env)
    assert returncode == 1, stderr
    assert str(proj_db) in stderr
    assert "Remove it" in stderr


def test_incompatible_proj_db_in_non_ascii_user_dir(env_without_proj_db, tmp_path):
    """A rejected proj.db is found in a directory with a non-ASCII name

    PROJ uses UTF-8 paths, which on Windows need a wide-character file
    check. The user directory is under the user's home directory there,
    so non-ASCII user names are a common case.
    """
    env, _, _ = env_without_proj_db
    user_dir = tmp_path / "user_dir_\u00e9"
    user_dir.mkdir()
    env["PROJ_USER_WRITABLE_DIRECTORY"] = str(user_dir)
    write_incompatible_proj_db(user_dir / "proj.db")
    returncode, stderr = run_g_proj(["-w", "epsg=3358"], env=env)
    assert returncode == 1, stderr
    # The path itself is not checked since its encoding in the output
    # depends on the platform.
    assert "Remove it" in stderr
