# MODULE:    grass.addons
#
# AUTHOR(S): Vaclav Petras <wenzeslaus gmail com>
#
# PURPOSE:   Addon (extension) management library
#
# COPYRIGHT: (C) 2026 by Vaclav Petras, and the GRASS Development Team
#
#            This program is free software under the GNU General Public
#            License (>=v2). Read the file COPYING that comes with GRASS
#            for details.

"""Exceptions used by the grass.addons package

The package reports all errors by raising exceptions derived from
:class:`AddonsError`; it never ends the process or prints messages.
Tools are expected to catch :class:`AddonsError` and translate it to
their fatal-error mechanism.
"""


class AddonsError(Exception):
    """Base class for all errors raised by the grass.addons package"""


class SourceResolutionError(AddonsError):
    """Addon source code could not be determined from the given name or URL"""


class FetchError(AddonsError):
    """Addon source code or binary could not be retrieved"""


class BuildError(AddonsError):
    """Addon compilation or documentation build failed"""


class InstallationError(AddonsError):
    """Addon files could not be installed"""


class RemovalError(AddonsError):
    """Addon files could not be removed"""


class RegistryError(AddonsError):
    """Local metadata files about installed addons could not be read or written"""


class CatalogError(AddonsError):
    """Metadata about available addons could not be retrieved or parsed"""


class MissingPrerequisiteError(AddonsError):
    """A program required for the requested operation is not available"""


class OperationCancelled(AddonsError):
    """The operation was cancelled through the reporter's cancellation check"""
