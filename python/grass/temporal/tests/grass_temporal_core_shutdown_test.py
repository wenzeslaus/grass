"""Test that a process using grass.temporal shuts down cleanly"""

import os
import subprocess
import sys

import grass.script as gs


def test_tgis_init_exits_cleanly(tmp_path):
    """Check that a script calling tgis.init() exits without errors.

    The temporal framework starts multiprocessing server processes and a
    watchdog thread which restarts them when they die. This is a regression
    test for the watchdog attempting a restart during interpreter shutdown,
    which fails loudly under the forkserver start method (the default on
    Linux since Python 3.14) because the forkserver machinery is already
    torn down at that point.
    """
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        result = subprocess.run(
            [sys.executable, "-c", "import grass.temporal as tgis; tgis.init()"],
            env=session.env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert "leaked semaphore" not in result.stderr, result.stderr
