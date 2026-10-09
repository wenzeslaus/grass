"""Tests of g.proj behavior when PROJ cannot find its database"""

import subprocess

import pytest

import grass.script as gs


@pytest.fixture
def env_without_proj_db(session, tmp_path):
    """Environment in which PROJ cannot find proj.db

    PROJ looks into the user-writable directory first and, when PROJ_DATA
    is set, only into the directories listed there, so pointing both to
    empty directories hides every proj.db on the machine.
    """
    env = session.env.copy()
    proj_data = tmp_path / "empty_proj_data"
    proj_data.mkdir()
    user_dir = tmp_path / "empty_user_dir"
    user_dir.mkdir()
    env["PROJ_DATA"] = str(proj_data)
    env.pop("PROJ_LIB", None)
    env["PROJ_USER_WRITABLE_DIRECTORY"] = str(user_dir)
    return env, proj_data


def test_epsg_works_with_proj_db(session):
    """Sanity check that the EPSG code resolves in a normal environment"""
    output = gs.read_command("g.proj", flags="w", epsg="3358", env=session.env)
    assert "North Carolina" in output


def test_epsg_fails_with_hint_without_proj_db(env_without_proj_db):
    """Without proj.db, g.proj fails with an error naming the searched paths

    The return code must be 1 (a GRASS fatal error), not a crash.
    """
    env, proj_data = env_without_proj_db
    process = gs.Popen(
        ["g.proj", "-w", "epsg=3358"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = process.communicate()
    assert process.returncode == 1, stderr
    assert "ERROR:" in stderr
    assert "proj.db" in stderr
    assert "PROJ_DATA" in stderr
    assert str(proj_data) in stderr, "The error should name the directory PROJ searched"
