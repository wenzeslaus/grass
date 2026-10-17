# MODULE:    grass.addons
#
# AUTHOR(S): Markus Neteler
#            Martin Landa <landa.martin gmail com>
#            Vaclav Petras <wenzeslaus gmail com>
#
# PURPOSE:   Addon (extension) management library
#
# COPYRIGHT: (C) 2009-2026 by Markus Neteler, and the GRASS Development Team
#
#            This program is free software under the GNU General Public
#            License (>=v2). Read the file COPYING that comes with GRASS
#            for details.

"""Defaults and installation-destination resolution for grass.addons"""

import os
import sys
from pathlib import Path

from grass.app.runtime import get_grass_config_dir_for_version

from .exceptions import InstallationError
from .reporter import NullReporter

OFFICIAL_REPOSITORY_URL = "https://github.com/OSGeo/grass-addons/"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def default_make_program():
    """Return the name of the GNU Make program for this platform

    GRASS Makefiles are of the GNU Make type, not BSD Make. On FreeBSD
    (and other BSD and maybe unix), the GNU Make program is usually
    called "gmake" to distinguish it from the (BSD) "make".
    """
    if sys.platform.startswith("freebsd"):
        return "gmake"
    return "make"


def resolve_install_prefix(
    path, to_system, *, major_version, minor_version, env, reporter=None
):
    """Determine and check the path for installation

    Returns the absolute path. Unlike the historical g.extension code,
    this does not modify the process environment; the caller is
    responsible for exporting the result (as GRASS_PREFIX_ADDON_BASE)
    when a subsequent build needs it.

    :raises InstallationError: when the path exists but is not writable
    """
    if reporter is None:
        reporter = NullReporter()
    if to_system:
        path = env["GISBASE"]
    if path == "$GRASS_ADDON_BASE":
        if not env.get("GRASS_ADDON_BASE"):
            path = os.path.join(
                get_grass_config_dir_for_version(major_version, minor_version, env=env),
                "addons",
            )
            reporter.warning(
                _("GRASS_ADDON_BASE is not defined, installing to {}").format(path)
            )
        else:
            path = env["GRASS_ADDON_BASE"]
    if Path(path).exists() and not os.access(path, os.W_OK):
        raise InstallationError(
            _(
                "You don't have permission to install extension to <{0}>."
                " Try to run {1} with administrator rights"
                " (su or sudo)."
            ).format(path, "g.extension")
        )
    return os.path.abspath(path)  # make likes absolute paths
