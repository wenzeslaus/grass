"""Temporary diagnostics: which Windows xfails come from the unshared environment?

Runs gunittest files with tests marked xfail_windows twice, each in a new
temporary mapset: as is, and with every change of os.environ also copied
into the environment of the C libraries (as if the environment were shared
as on other platforms). A test which passes only in the second run fails
because the C libraries do not see variables set in the running process.

Usage:

    python -u diagnose_xfail_windows.py <grass command> <project> <file>...
    python -u diagnose_xfail_windows.py --run <mode> <file>
"""

import faulthandler
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def share_environment():
    """Copy each later change of os.environ into the C libraries"""
    import grass.lib.gis as libgis

    original_putenv = os.putenv
    original_unsetenv = os.unsetenv

    def encode(text):
        if isinstance(text, bytes):
            return text
        return text.encode("mbcs") if sys.platform == "win32" else os.fsencode(text)

    def putenv(name, value):
        original_putenv(name, value)
        libgis.G_putenv(encode(name), encode(value))

    def unsetenv(name):
        original_unsetenv(name)
        libgis.G_putenv(encode(name), b"")

    # os.environ calls these module functions on each change.
    os.putenv = putenv
    os.unsetenv = unsetenv


def run_test_file(mode, path):
    if mode == "shared":
        share_environment()
    faulthandler.dump_traceback_later(150, exit=True)
    sys.argv = [path]
    sys.path.insert(0, str(Path(path).parent))
    runpy.run_path(path, run_name="__main__")


def main(grass, project, paths):
    for path in paths:
        path = Path(path).resolve()
        for mode in ("as-is", "shared"):
            start = time.monotonic()
            try:
                process = subprocess.run(
                    [
                        grass,
                        "--tmp-mapset",
                        project,
                        "--exec",
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--run",
                        mode,
                        str(path),
                    ],
                    cwd=path.parent,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=240,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                print(f"== {path.name} {mode}: TIMEOUT")
                continue
            lines = process.stderr.strip().splitlines()
            summary = [
                line
                for line in lines
                if line.startswith(("Ran ", "OK", "FAILED", "Timeout"))
                or "Error" in line
            ]
            duration = time.monotonic() - start
            print(
                f"== {path.name} {mode} (exit code {process.returncode}, {duration:.0f} s)"
            )
            for line in summary[-8:]:
                print(f"  {line[:300]}")


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run_test_file(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
