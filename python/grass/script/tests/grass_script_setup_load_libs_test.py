# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of loading the C libraries into the process for grass.lib

This file is meant to be executed also standalone (pytest with this file as
the only argument) against an installation which was moved away from the
prefix it was configured with. Only then the GRASS libraries cannot find each
other through their RUNPATH and the tests actually fail without the loading.
"""

import json
import os
import subprocess
import sys
from textwrap import dedent

import pytest

import grass.app.runtime
import grass.script as gs

WINDOWS = sys.platform.startswith("win")


def run_in_clean_environment(code, tmp_path):
    """Run code in a subprocess without the dynamic library search path variable

    The variable is what a parent GRASS session or a manual setup uses to make
    the GRASS libraries available, so removing it leaves the loading done on
    the grass.lib import as the only mechanism which can make grass.lib work.
    """
    source_file = tmp_path / "code.py"
    source_file.write_text(dedent(code))
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("DYLD_LIBRARY_PATH", None)
    result = subprocess.run(
        [sys.executable, os.fspath(source_file)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        result.stderr
        or f"Exit code {result.returncode} without any output (a fatal error in"
        " the GRASS C libraries ends the process silently when they cannot see"
        " GISBASE)"
    )
    return json.loads(result.stdout)


@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_importable(tmp_path):
    """Check that grass.lib modules can be imported after init

    The import loads the libraries, which fails when the GRASS libraries they
    depend on cannot be found.
    """
    project = tmp_path / "test"
    code = f"""
        import json
        import grass.script as gs

        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}"):
            import grass.lib.gis
            import grass.lib.raster
        print(json.dumps({{"imported": True}}))
    """
    assert run_in_clean_environment(code, tmp_path=tmp_path)["imported"]


@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_importable_without_session(tmp_path):
    """Check that grass.lib can be imported without any session

    Importing only loads the libraries; a session is needed for calling
    functions which use it, not for the import.
    """
    code = """
        import json
        import os

        os.environ.pop("GISBASE", None)
        import grass.lib.gis
        import grass.lib.raster
        print(json.dumps({"imported": True}))
    """
    assert run_in_clean_environment(code, tmp_path=tmp_path)["imported"]


@pytest.mark.xfail(
    WINDOWS,
    reason="On Windows, the C libraries (built with a different C runtime than"
    " Python) do not see the environment variables set by init in the running"
    " process, so they do not see the session",
)
@pytest.mark.usefixtures("mock_no_session")
def test_grass_lib_usable(tmp_path):
    """Check that grass.lib sees the session and works with rasters after init

    Opening a raster needs the session state held by libgrass_gis, so this
    also fails when the process ends up with two copies of that library.
    """
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
    assert run_in_clean_environment(code, tmp_path=tmp_path)["raster_opened"]


@pytest.mark.usefixtures("mock_no_session")
def test_init_does_not_load_libraries(tmp_path, monkeypatch):
    """Check that a session does not load the C libraries

    The loading happens on the grass.lib import, so sessions using only
    tools don't pay for it.
    """
    calls = []

    def record_call(install_path):
        calls.append(install_path)
        # The replaced function returns the list of load failures.
        return []

    monkeypatch.setattr(grass.app.runtime, "preload_dynamic_libraries", record_call)
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()):
        pass
    assert not calls


@pytest.mark.skipif(WINDOWS, reason="Loading is a no-op on Windows")
def test_grass_lib_load_failure_raises_import_error(tmp_path):
    """Check that a failure to load the core library is an import error"""
    import grass.lib

    lib_path = tmp_path / "lib"
    lib_path.mkdir()
    (lib_path / "libgrass_gis.9.9.so").write_text("This is not a shared library.")
    with pytest.raises(ImportError, match="libgrass_gis"):
        grass.lib._load_libraries(tmp_path)


def test_grass_lib_loading_without_grass_libraries_is_noop(tmp_path):
    """Check that loading does nothing without the core library files

    This is the case on Windows, where the library files have different names
    and the loading is left to the loader of the wrapper modules.
    """
    import grass.lib

    lib_path = tmp_path / "lib"
    lib_path.mkdir()
    (lib_path / "libgrass_notalibrary.so").write_text("This is not a shared library.")
    grass.lib._load_libraries(tmp_path)


def test_preload_reports_libraries_which_cannot_be_loaded(tmp_path):
    """Check that libraries which fail to load are reported, not raised"""
    lib_path = tmp_path / "lib"
    lib_path.mkdir()
    broken = lib_path / "libgrass_notalibrary.so"
    broken.write_text("This is not a shared library.")
    failures = grass.app.runtime.preload_dynamic_libraries(install_path=tmp_path)
    if WINDOWS:
        assert failures == []
    else:
        # Paths are compared by name because the function resolves them and
        # the temporary directory may be behind a symbolic link.
        assert [path.name for path, _error in failures] == [broken.name]
