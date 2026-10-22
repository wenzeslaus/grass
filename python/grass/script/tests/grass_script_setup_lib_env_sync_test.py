# SPDX-FileCopyrightText: 2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of the C libraries (grass.lib) seeing a session set up in their process

On Windows, the C libraries may keep their own copy of the environment, so
grass.lib copies the session variables into it when it is imported, init
copies them again when grass.lib is already imported, and
grass.lib.sync_environment copies them on request. On other platforms, the
environment is shared and the tests check the same behavior without the
copying.
"""

import pytest


@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_sees_session_of_init_in_same_process(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Check that grass.lib imported after init sees the session"""
    project = tmp_path / "test"
    code = f"""
        import json

        import grass.script as gs

        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}"):
            import grass.lib.gis as libgis
            import grass.lib.raster as libraster

            libgis.G_gisinit(b"test")
            gs.run_command("g.region", rows=2, cols=2)
            gs.mapcalc("ones = 1")
            fd = libraster.Rast_open_old(b"ones", b"")
            libraster.Rast_close(fd)
        print(json.dumps({{"raster_opened": True}}))
    """
    assert run_code_in_environment(code, env=clean_subprocess_environment)[
        "raster_opened"
    ]


@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_imported_before_init_sees_session(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Check that grass.lib imported before init sees the session"""
    project = tmp_path / "test"
    code = f"""
        import json
        import os

        import grass.lib.gis as libgis
        import grass.lib.raster as libraster
        import grass.script as gs

        gisbase_before = os.environ.get("GISBASE")
        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}"):
            libgis.G_gisinit(b"test")
            gs.run_command("g.region", rows=2, cols=2)
            gs.mapcalc("ones = 1")
            fd = libraster.Rast_open_old(b"ones", b"")
            libraster.Rast_close(fd)
        print(json.dumps({{"raster_opened": True, "gisbase_before": gisbase_before}}))
    """
    result = run_code_in_environment(code, env=clean_subprocess_environment)
    assert result["raster_opened"]
    assert result["gisbase_before"] is None


@pytest.mark.usefixtures("mock_no_session")
def test_grass_variables_reach_c_libraries(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Check that a GRASS_ variable set in the process is seen by the C libraries"""
    project = tmp_path / "test"
    code = f"""
        import json
        import os

        import grass.script as gs

        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}"):
            os.environ["GRASS_VERBOSE"] = "3"
            import grass.lib.gis as libgis

            libgis.G_gisinit(b"test")
            verbosity = libgis.G_verbose()
        print(json.dumps({{"verbosity": verbosity}}))
    """
    assert (
        run_code_in_environment(code, env=clean_subprocess_environment)["verbosity"]
        == 3
    )


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
