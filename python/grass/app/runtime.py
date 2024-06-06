"""Provides functions for the main GRASS GIS executable

(C) 2020 by Vaclav Petras and the GRASS Development Team

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

.. sectionauthor:: Vaclav Petras <wenzeslaus gmail com>
.. sectionauthor:: Linda Kladivova <l.kladivova seznam cz>

This is not a stable part of the API. Use at your own risk.
"""

import os
import subprocess
import sys
import shutil
import collections

# Get the system name
WINDOWS = sys.platform.startswith("win")
CYGWIN = sys.platform.startswith("cygwin")
MACOS = sys.platform.startswith("darwin")

GISBASE = None


def set_gisbase(path, /):
    global GISBASE
    GISBASE = path


def gpath(*args):
    """Construct path to file or directory in GRASS GIS installation

    Can be called only after GISBASE was set.
    """
    return os.path.join(GISBASE, *args)


def wxpath(*args):
    """Construct path to file or directory in GRASS wxGUI

    Can be called only after GISBASE was set.

    This function does not check if the directories exist or if GUI works
    this must be done by the caller if needed.
    """
    global _WXPYTHON_BASE
    if not _WXPYTHON_BASE:
        # this can be called only after GISBASE was set
        _WXPYTHON_BASE = gpath("gui", "wxpython")
    return os.path.join(_WXPYTHON_BASE, *args)


def path_prepend(directory, var):
    path = os.getenv(var)
    if path:
        path = directory + os.pathsep + path
    else:
        path = directory
    os.environ[var] = path


def get_grass_config_dir(major_version, minor_version, env):
    """Get configuration directory

    Determines path of GRASS GIS user configuration directory.
    """
    # The code is in sync with grass.app.runtime (but not the same).
    if WINDOWS:
        config_dirname = f"GRASS{major_version}"
        return os.path.join(env.get("APPDATA"), config_dirname)
    elif MACOS:
        version = f"{major_version}.{minor_version}"
        return os.path.join(env.get("HOME"), "Library", "GRASS", version)
    else:
        config_dirname = f".grass{major_version}"
        return os.path.join(env.get("HOME"), config_dirname)


def append_left_main_executable_paths(paths, install_path):
    # define PATH
    paths.appendleft(os.path.join(install_path, "bin"))
    paths.appendleft(os.path.join(install_path, "scripts"))
    if WINDOWS:
        paths.appendleft(os.path.join(install_path, "extrabin"))


def append_left_addon_paths(paths, config_dir, env):
    # addons (base)
    addon_base = env.get("GRASS_ADDON_BASE")
    if not addon_base:
        if MACOS:
            name = "Addons"
        else:
            name = "addons"
        addon_base = os.path.join(config_dir, name)
        env["GRASS_ADDON_BASE"] = addon_base

    if not WINDOWS:
        paths.appendleft(os.path.join(addon_base, "scripts"))
    paths.appendleft(os.path.join(addon_base, "bin"))

    # addons (path)
    addon_path = env.get("GRASS_ADDON_PATH")
    if addon_path:
        for path in addon_path.split(os.pathsep):
            paths.appendleft(path)


def set_executable_paths(install_path, grass_config_dir, env):
    paths = collections.deque()
    # Addons
    append_left_addon_paths(paths, grass_config_dir, env=env)
    # Standard installation
    append_left_main_executable_paths(paths, install_path=install_path)

    paths.append(env.get("PATH"))
    env["PATH"] = os.pathsep.join(paths)


def set_paths(grass_config_dir, ld_library_path_variable_name):
    set_executable_paths(
        install_path=GISBASE, grass_config_dir=grass_config_dir, env=os.environ
    )
    set_python_path_variable(install_path=GISBASE, env=os.environ)

    # set path for the GRASS man pages
    grass_man_path = gpath("docs", "man")
    # retrieving second time, but now it is always set
    addon_base = os.getenv("GRASS_ADDON_BASE")
    addons_man_path = os.path.join(addon_base, "docs", "man")
    man_path = os.getenv("MANPATH")
    if man_path:
        path_prepend(addons_man_path, "MANPATH")
        path_prepend(grass_man_path, "MANPATH")
    else:
        sys_man_path = None
        manpath_executable = shutil.which("manpath")
        if manpath_executable:
            try:
                sys_man_path = subprocess.run(
                    manpath_executable,
                    text=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass

        if sys_man_path:
            os.environ["MANPATH"] = sys_man_path
            path_prepend(addons_man_path, "MANPATH")
            path_prepend(grass_man_path, "MANPATH")
        else:
            os.environ["MANPATH"] = addons_man_path
            path_prepend(grass_man_path, "MANPATH")

    # Set LD_LIBRARY_PATH (etc) to find GRASS shared libraries
    # this works for subprocesses but won't affect the current process
    if ld_library_path_variable_name:
        set_dynamic_library_path(
            variable_name=ld_library_path_variable_name,
            install_path=GISBASE,
            env=os.environ,
        )


def set_dynamic_library_path(variable_name, install_path, env):
    # define LD_LIBRARY_PATH
    if variable_name not in env:
        env[variable_name] = ""
    env[variable_name] += os.pathsep + os.path.join(install_path, "lib")


def set_python_path_variable(install_path, env):
    """Set PYTHONPATH to find GRASS Python package in subprocesses"""
    path = env.get("PYTHONPATH")
    etcpy = os.path.join(install_path, "etc", "python")
    if path:
        path = etcpy + os.pathsep + path
    else:
        path = etcpy
    env["PYTHONPATH"] = path


def find_exe(pgm):
    for directory in os.getenv("PATH").split(os.pathsep):
        path = os.path.join(directory, pgm)
        if os.access(path, os.X_OK):
            return path
    return None


def set_path_to_python_executable(env):
    # Set GRASS_PYTHON
    if not env.get("GRASS_PYTHON"):
        if WINDOWS:
            env["GRASS_PYTHON"] = "python3.exe"
        else:
            env["GRASS_PYTHON"] = "python3"


def set_defaults(config_projshare_path):
    # GRASS_PAGER
    if not os.getenv("GRASS_PAGER"):
        if find_exe("more"):
            pager = "more"
        elif find_exe("less"):
            pager = "less"
        elif WINDOWS:
            pager = "more"
        else:
            pager = "cat"
        os.environ["GRASS_PAGER"] = pager

    # GRASS_PYTHON
    set_path_to_python_executable(env=os.environ)

    # GRASS_GNUPLOT
    if not os.getenv("GRASS_GNUPLOT"):
        os.environ["GRASS_GNUPLOT"] = "gnuplot -persist"

    # GRASS_PROJSHARE
    if not os.getenv("GRASS_PROJSHARE") and config_projshare_path:
        os.environ["GRASS_PROJSHARE"] = config_projshare_path


def set_display_defaults():
    """Predefine monitor size for certain architectures"""
    if os.getenv("HOSTTYPE") == "arm":
        # small monitor on ARM (iPAQ, zaurus... etc)
        os.environ["GRASS_RENDER_HEIGHT"] = "320"
        os.environ["GRASS_RENDER_WIDTH"] = "240"


def set_browser():
    # GRASS_HTML_BROWSER
    browser = os.getenv("GRASS_HTML_BROWSER")
    if not browser:
        if MACOS:
            # OSX doesn't execute browsers from the shell PATH - route through a
            # script
            browser = gpath("etc", "html_browser_mac.sh")
            os.environ["GRASS_HTML_BROWSER_MACOSX"] = "-b com.apple.helpviewer"

        if WINDOWS:
            browser = "start"
        elif CYGWIN:
            browser = "explorer"
        else:
            # the usual suspects
            browsers = [
                "xdg-open",
                "x-www-browser",
                "htmlview",
                "konqueror",
                "mozilla",
                "mozilla-firefox",
                "firefox",
                "iceweasel",
                "opera",
                "google-chrome",
                "chromium",
                "netscape",
                "dillo",
                "lynx",
                "links",
                "w3c",
            ]
            for b in browsers:
                if find_exe(b):
                    browser = b
                    break

    elif MACOS:
        # OSX doesn't execute browsers from the shell PATH - route through a
        # script
        os.environ["GRASS_HTML_BROWSER_MACOSX"] = "-b %s" % browser
        browser = gpath("etc", "html_browser_mac.sh")

    if not browser:
        # even so we set to 'xdg-open' as a generic fallback
        browser = "xdg-open"

    os.environ["GRASS_HTML_BROWSER"] = browser


def ensure_home():
    """Set HOME if not set on MS Windows"""
    if WINDOWS and not os.getenv("HOME"):
        os.environ["HOME"] = os.path.join(os.getenv("HOMEDRIVE"), os.getenv("HOMEPATH"))
