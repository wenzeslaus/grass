# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of the C libraries (grass.lib) seeing a session set up in their process

On Windows, the C libraries may keep their own copy of the environment, which
grass.lib updates with the session variables. On other platforms, the
environment is shared and the tests check the same behavior without the
copying.
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


def test_grass_lib_importable_with_variable_outside_code_page(
    run_code_in_environment, clean_subprocess_environment
):
    """Check that a GRASS_ variable which cannot be copied does not break the import

    On Windows, a value outside of the ANSI code page cannot be passed to the
    C runtime of the libraries.
    """
    env = clean_subprocess_environment
    env["GRASS_TEST_VARIABLE"] = "\u4e2d\u6587"
    code = """
        import json

        import grass.lib.gis

        print(json.dumps({"imported": True}))
    """
    assert run_code_in_environment(code, env=env)["imported"]
