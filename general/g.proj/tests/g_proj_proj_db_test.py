# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: 2026 GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests of g.proj behavior when PROJ cannot use its database"""

from pathlib import Path

import grass.script as gs
from grass.tools import Tools


def collapse_whitespace(text):
    """Collapse whitespace to single spaces to search in wrapped messages"""
    return " ".join(text.split())


def test_print_crs_without_proj_db(tools_without_proj_db):
    """The CRS of the current project is printed without proj.db"""
    result = tools_without_proj_db.g_proj(flags="p")
    assert result.returncode == 0, result.stderr
    assert "EPSG" in result.text


def test_interface_description_without_proj_db(tools_without_proj_db):
    """The interface description is available without proj.db"""
    result = tools_without_proj_db.run_cmd(["g.proj", "--interface-description"])
    assert result.returncode == 0, result.stderr
    assert "list_codes" in result.text


def test_list_codes_fails_without_proj_db(tools_without_proj_db):
    """Listing codes ends with an error, not a crash, without proj.db"""
    result = tools_without_proj_db.g_proj(list_codes="EPSG")
    assert result.returncode == 1, result.stderr


def test_epsg_works_with_proj_db(session):
    """Sanity check that the EPSG code resolves in a normal environment"""
    output = gs.read_command("g.proj", flags="w", epsg="3358", env=session.env)
    assert "North Carolina" in output


def test_epsg_fails_with_hint_without_proj_db(
    env_without_proj_db, tools_without_proj_db
):
    """Without proj.db, g.proj fails with a message naming the searched paths

    The return code must be 1 (a GRASS fatal error), not a crash.
    """
    result = tools_without_proj_db.g_proj(flags="w", epsg=3358)
    assert result.returncode == 1, result.stderr
    stderr = collapse_whitespace(result.stderr)
    assert "proj.db" in stderr
    assert "PROJ_DATA" in stderr or "PROJ_LIB" in stderr
    assert env_without_proj_db["PROJ_DATA"] in stderr, (
        "The message should name the searched path"
    )


def test_epsg_fails_with_hint_for_incompatible_proj_db(
    env_without_proj_db, tools_without_proj_db, write_incompatible_proj_db
):
    """A proj.db which PROJ rejects is reported as found but unusable

    This is the case of PROJ_DATA pointing to a database from another
    PROJ installation, simulated by a database with a future layout.
    """
    proj_db = Path(env_without_proj_db["PROJ_DATA"]) / "proj.db"
    write_incompatible_proj_db(proj_db)
    result = tools_without_proj_db.g_proj(flags="w", epsg=3358)
    assert result.returncode == 1, result.stderr
    stderr = collapse_whitespace(result.stderr)
    assert "another PROJ installation" in stderr
    assert str(proj_db) in stderr, "The message should name the rejected file"


def test_epsg_fails_with_hint_for_proj_lib_only(env_without_proj_db):
    """When only PROJ_LIB is set, the message names PROJ_LIB and its value"""
    env = env_without_proj_db
    del env["PROJ_DATA"]
    tools = Tools(env=env, errors="ignore", consistent_return_value=True)
    result = tools.g_proj(flags="w", epsg=3358)
    assert result.returncode == 1, result.stderr
    stderr = collapse_whitespace(result.stderr)
    assert f"PROJ_LIB is set to <{env['PROJ_LIB']}>" in stderr


def test_epsg_fails_with_hint_for_incompatible_proj_db_in_user_dir(
    env_without_proj_db, tools_without_proj_db, write_incompatible_proj_db
):
    """A rejected proj.db in the PROJ user directory is to be removed"""
    proj_db = Path(env_without_proj_db["PROJ_USER_WRITABLE_DIRECTORY"]) / "proj.db"
    write_incompatible_proj_db(proj_db)
    result = tools_without_proj_db.g_proj(flags="w", epsg=3358)
    assert result.returncode == 1, result.stderr
    stderr = collapse_whitespace(result.stderr)
    assert str(proj_db) in stderr
    assert "Remove it" in stderr


def test_incompatible_proj_db_in_non_ascii_user_dir(
    env_without_proj_db, write_incompatible_proj_db, tmp_path
):
    """A rejected proj.db is found in a directory with a non-ASCII name

    The PROJ user directory is under the user's home directory, so
    non-ASCII user names are a common case. On Windows, PROJ keeps the path
    from the environment variable used here in the ANSI code page, not in
    UTF-8.
    """
    env = env_without_proj_db
    user_dir = tmp_path / "user_dir_é"
    user_dir.mkdir()
    env["PROJ_USER_WRITABLE_DIRECTORY"] = str(user_dir)
    write_incompatible_proj_db(user_dir / "proj.db")
    # grass.tools fails to decode the path in the ANSI code page on Windows.
    process = gs.Popen(
        ["g.proj", "-w", "epsg=3358"],
        env=env,
        stdout=gs.PIPE,
        stderr=gs.PIPE,
        text=True,
        errors="replace",
    )
    _, stderr = process.communicate()
    assert process.returncode == 1, stderr
    # The path itself is not checked since its encoding in the output
    # depends on the platform.
    assert "Remove it" in collapse_whitespace(stderr)
