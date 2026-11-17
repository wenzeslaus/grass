# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of the C libraries (grass.lib) seeing a session set up in their process

On Windows, the C libraries may keep their own copy of the environment, which
grass.lib updates with the session variables. On other platforms, the
environment is shared and the tests check the same behavior without the
copying. With a custom env, sync_environment copies the variables on
all platforms.
"""

import pytest


@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_imported_after_init_sees_session_and_variables(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Check that grass.lib imported after init sees what the process has set"""
    project = tmp_path / "test"
    code = f"""
        import json
        import os

        import grass.script as gs

        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}"):
            os.environ["GRASS_VERBOSE"] = "3"
            import grass.lib.gis as libgis
            import grass.lib.raster as libraster

            libgis.G_gisinit(b"test")
            verbosity = libgis.G_verbose()
            gs.run_command("g.region", rows=2, cols=2)
            gs.mapcalc("ones = 1")
            fd = libraster.Rast_open_old(b"ones", b"")
            libraster.Rast_close(fd)
        print(json.dumps({{"raster_opened": True, "verbosity": verbosity}}))
    """
    assert run_code_in_environment(code, env=clean_subprocess_environment) == {
        "raster_opened": True,
        "verbosity": 3,
    }


@pytest.mark.usefixtures("mock_no_session")
def test_explicit_sync_updates_and_removes_variables(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Check that sync_environment passes on changed and removed variables

    GISBASE is used because libgis reads it directly from the environment
    of its C runtime on every call.
    """
    project = tmp_path / "test"
    code = f"""
        import json
        import os

        import grass.lib
        import grass.lib.gis as libgis
        import grass.script as gs


        def seen_by_c():
            value = libgis.G_getenv_nofatal(b"GISBASE")
            return None if value is None else os.fsdecode(value)


        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}"):
            gisbase = os.environ["GISBASE"]
            os.environ["GISBASE"] = "changed"
            grass.lib.sync_environment()
            after_change = seen_by_c()
            del os.environ["GISBASE"]
            grass.lib.sync_environment()
            after_removal = seen_by_c()
            os.environ["GISBASE"] = gisbase
            grass.lib.sync_environment()
        print(json.dumps({{"after_change": after_change, "after_removal": after_removal}}))
    """
    assert run_code_in_environment(code, env=clean_subprocess_environment) == {
        "after_change": "changed",
        "after_removal": None,
    }


@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_sees_custom_env_session_until_sync_without_env(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Check a custom env session with sync_environment with and without env"""
    project = tmp_path / "test"
    code = f"""
        import json
        import os

        import grass.lib
        import grass.lib.gis as libgis
        import grass.lib.raster as libraster
        import grass.script as gs

        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}", env=os.environ.copy()) as session:
            gs.run_command("g.region", rows=2, cols=2, env=session.env)
            gs.mapcalc("ones = 1", env=session.env)
            grass.lib.sync_environment(env=session.env)
            libgis.G_gisinit(b"test")
            fd = libraster.Rast_open_old(b"ones", b"")
            libraster.Rast_close(fd)
            gisrc_in_os_environ = "GISRC" in os.environ
        grass.lib.sync_environment()
        gisbase_seen_after = libgis.G_getenv_nofatal(b"GISBASE") is not None
        print(
            json.dumps(
                {{
                    "raster_opened": True,
                    "gisrc_in_os_environ": gisrc_in_os_environ,
                    "gisbase_seen_after": gisbase_seen_after,
                }}
            )
        )
    """
    assert run_code_in_environment(code, env=clean_subprocess_environment) == {
        "raster_opened": True,
        "gisrc_in_os_environ": False,
        "gisbase_seen_after": False,
    }


def test_grass_lib_importable_with_variable_outside_code_page(
    run_code_in_environment, clean_subprocess_environment
):
    """Check that a GRASS_ variable which cannot be copied does not break the import

    On Windows, a value outside of the ANSI code page cannot be passed to the
    C runtime of the libraries. Processes started afterwards still get it.
    """
    env = clean_subprocess_environment
    env["GRASS_TEST_VARIABLE"] = "\u4e2d\u6587"
    code = """
        import json
        import subprocess
        import sys

        import grass.lib.gis

        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; print(ascii(os.environ.get('GRASS_TEST_VARIABLE')))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(json.dumps({"imported": True, "in_child": child.stdout.strip()}))
    """
    assert run_code_in_environment(code, env=env) == {
        "imported": True,
        "in_child": ascii("\u4e2d\u6587"),
    }
