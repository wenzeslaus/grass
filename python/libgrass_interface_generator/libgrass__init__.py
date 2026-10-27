import os

from .ctypes_loader import add_library_search_dirs


def _core_libraries(lib_path):
    """Get the real paths of the core shared library files in a directory"""
    try:
        names = os.listdir(lib_path)
    except OSError:
        return []
    return sorted(
        {
            os.path.realpath(os.path.join(lib_path, name))
            for name in names
            if name.startswith(("libgrass_gis.", "grass_gis."))
            and (name.endswith((".so", ".dylib", ".dll")) or ".so." in name)
        }
    )


def _find_lib_path():
    """Find the directory with the GRASS libraries this package belongs to.

    The directory is derived from the location of this file, which also
    guarantees that the libraries match the wrapper modules generated for
    them. The package is at <install path>/etc/python/grass/lib with the
    libraries in <install path>/lib, or, in an FHS layout, in site-packages
    under the library directory, e.g., <prefix>/lib/python3.12/site-packages.
    GISBASE is used as a fallback for other layouts.
    """
    path = os.path.dirname(os.path.realpath(__file__))
    # Four levels up from grass/lib is the installation in the traditional
    # layout (lib, grass, python, etc) and the library directory in the FHS
    # layout (lib, grass, site-packages, python3.X). For a shallower path, the
    # climb stops at the filesystem root, where no candidate matches.
    for _ in range(4):
        path = os.path.dirname(path)
    candidates = [os.path.join(path, "lib"), path]
    gisbase = os.environ.get("GISBASE")
    if gisbase:
        candidates.append(os.path.join(gisbase, "lib"))
    for candidate in candidates:
        if _core_libraries(candidate):
            return candidate
    return None


_lib_path = _find_lib_path()
if _lib_path:
    # Registering the directory makes the wrapper modules resolve the
    # libraries by absolute path, skipping the expensive
    # ctypes.util.find_library() fallback.
    add_library_search_dirs([_lib_path])
    if hasattr(os, "add_dll_directory"):
        # On Windows, dependent DLLs resolve from the directories registered
        # here and from the directory of the loaded DLL itself.
        os.add_dll_directory(_lib_path)

__all__ = [
    'arraystats',
    'cluster',
    'date',
    'dbmi',
    'display',
    'gis',
    'gmath',
    'imagery',
    'nviz',
    'ogsf',
    'proj',
    'raster3d',
    'raster',
    'rtree',
    'stats',
    'vector',
    'vedit'
]
