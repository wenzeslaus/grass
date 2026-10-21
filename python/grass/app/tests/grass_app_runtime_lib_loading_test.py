# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of making the GRASS C libraries (grass.lib) work without a session

The loading done by the grass.lib package on import is tested here through
its private functions because grass.lib is generated and has no tests
directory of its own.

This file is meant to be executed also standalone (pytest with this file as
the only argument), including against an installation moved away from the
prefix it was configured with: the tests provide any needed environment
themselves instead of relying on the environment of the test process.
"""

import os
import sys
from pathlib import Path

import pytest

from grass.app.runtime import RuntimePaths
from grass.script.setup import get_install_path


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Loading is a no-op on Windows"
)
def test_grass_lib_load_failure_raises_import_error(tmp_path):
    """Check that a failure to load the core library is an import error"""
    import grass.lib

    lib_path = tmp_path / "lib"
    lib_path.mkdir()
    (lib_path / "libgrass_gis.so").write_text("This is not a shared library.")
    with pytest.raises(ImportError, match="libgrass_gis"):
        grass.lib._load_libraries(lib_path)


def test_grass_lib_loading_without_grass_libraries_is_noop(tmp_path, monkeypatch):
    """Check that loading does nothing without the core library files"""
    import grass.lib

    calls = []

    def record_call(lib_path):
        calls.append(lib_path)
        # The replaced function returns the list of load failures.
        return []

    monkeypatch.setattr("grass.lib._preload_libraries", record_call)
    lib_path = tmp_path / "lib"
    lib_path.mkdir()
    (lib_path / "libgrass_notalibrary.so").write_text("This is not a shared library.")
    grass.lib._load_libraries(lib_path)
    assert not calls


def test_preload_reports_libraries_which_cannot_be_loaded(tmp_path):
    """Check that libraries which fail to load are reported, not raised"""
    import grass.lib

    lib_path = tmp_path / "lib"
    lib_path.mkdir()
    broken = lib_path / "libgrass_notalibrary.so"
    broken.write_text("This is not a shared library.")
    failures = grass.lib._preload_libraries(lib_path)
    # Paths are compared by name because the function resolves them and
    # the temporary directory may be behind a symbolic link.
    assert [os.path.basename(path) for path, _error in failures] == [broken.name]


def test_grass_lib_importable_without_session(
    run_code_in_environment, clean_subprocess_environment
):
    """Check that grass.lib can be imported without any session or variable"""
    code = """
        import json

        import grass.lib.gis
        import grass.lib.raster

        print(json.dumps({"imported": True}))
    """
    assert run_code_in_environment(code, env=clean_subprocess_environment)["imported"]


def test_grass_lib_importable_with_manual_search_path(
    run_code_in_environment, clean_subprocess_environment
):
    """Check that the search path variable set at process start is enough"""
    lib_path = os.fspath(Path(get_install_path()) / "lib")
    env = clean_subprocess_environment
    variable = RuntimePaths().ld_library_path_var
    if variable in env:
        env[variable] = lib_path + os.pathsep + env[variable]
    else:
        env[variable] = lib_path
    if sys.platform == "darwin":
        # The variable GRASS knows on macOS (LD_RUN_PATH) is read by the
        # ctypes loader but ignored by dyld, which reads DYLD_LIBRARY_PATH.
        env["DYLD_LIBRARY_PATH"] = lib_path
    code = """
        import json

        import grass.lib.gis
        import grass.lib.raster

        print(json.dumps({"imported": True}))
    """
    assert run_code_in_environment(code, env=env)["imported"]
