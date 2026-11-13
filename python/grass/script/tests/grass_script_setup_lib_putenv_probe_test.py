"""Experiment: do child processes inherit what G_putenv sets or removes?"""

import json

import pytest


@pytest.mark.usefixtures("mock_no_session")
def test_probe_child_environment(
    tmp_path, run_code_in_environment, clean_subprocess_environment
):
    """Report what a child started with env=None sees (always fails to show it)"""
    project = tmp_path / "test"
    env = dict(clean_subprocess_environment)
    env["GRASS_PROBE_ASCII"] = "plain"
    env["GRASS_PROBE_OUTSIDE"] = "中文"
    code = f"""
        import json
        import os
        import subprocess
        import sys

        def child():
            code = (
                "import json, os; print(json.dumps({{name: os.environ.get(name)"
                " for name in ['GISRC', 'GRASS_PROBE_ASCII', 'GRASS_PROBE_OUTSIDE',"
                " 'GRASS_PROBE_SET_LATER']}}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            )
            return json.loads(result.stdout)

        results = {{"before_import": child()}}
        import grass.lib
        import grass.script as gs

        results["shared"] = grass.lib._is_environment_shared(grass.lib._libgis())
        results["after_import"] = child()
        os.environ["GRASS_PROBE_SET_LATER"] = "中"
        grass.lib.sync_environment()
        results["after_sync_with_unencodable_set_later"] = child()
        gs.create_project(r"{project}")
        with gs.setup.init(r"{project}", env=os.environ.copy()) as session:
            results["custom_env_before_sync"] = child()
            grass.lib.sync_environment(env=session.env)
            results["custom_env_after_sync"] = child()
        grass.lib.sync_environment()
        results["after_plain_sync"] = child()
        print(json.dumps({{"results": results}}))
    """
    results = run_code_in_environment(code, env=env)["results"]
    pytest.fail(json.dumps(results, indent=1))
