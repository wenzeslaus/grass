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

"""Helpers for tool (module) names and their class (family) names"""


def expand_module_class_name(class_letters):
    """Convert module class (family) letter or letters to class (family) name

    The letter or letters are used in module names, e.g. r.slope.aspect.
    The names are used in directories in Addons but also in the source code.

    >>> expand_module_class_name("r")
    'raster'
    >>> expand_module_class_name("v")
    'vector'
    """
    name = {
        "d": "display",
        "db": "db",
        "g": "general",
        "i": "imagery",
        "m": "misc",
        "ps": "postscript",
        "p": "paint",
        "r": "raster",
        "r3": "raster3d",
        "s": "sites",
        "t": "temporal",
        "v": "vector",
        "wx": "gui/wxpython",
    }

    return name.get(class_letters, class_letters)


def get_module_class_name(module_name):
    """Return class (family) name for a module

    The names are used in directories in Addons but also in the source code.

    >>> get_module_class_name("r.slope.aspect")
    'raster'
    >>> get_module_class_name("v.to.rast")
    'vector'
    """
    classchar = module_name.split(".", 1)[0]
    return expand_module_class_name(classchar)
