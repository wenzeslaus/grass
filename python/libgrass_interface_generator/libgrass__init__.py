import ctypes
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


# Every other GRASS library needs libgrass_gis and libgrass_gis needs
# libgrass_datetime, so these two are loaded before the rest. A library loaded
# before the GRASS libraries it needs makes the dynamic linker resolve those
# through the process-startup search path, which may find another GRASS
# installation and leave the process with two copies of the same library.
# Only these two are ordered: a second copy of libgrass_gis breaks every other
# library because they all rely on the session state it holds.
_BASE_LIBRARY_NAMES = ("datetime", "gis")


def _preload_libraries(lib_path):
    """Load the GRASS libraries in a directory into the process by full path.

    Libraries can be loaded only after the libraries they depend on, so
    loading is repeated until it makes no progress. Windows DLLs are not
    matched because dependent DLLs resolve from the registered directories.

    Returns a list of (path, error) pairs for libraries which failed to load.
    """
    # Versioned files and the unversioned symlinks pointing to them are both
    # listed, so the real paths are collected to load each file once.
    libraries = {
        os.path.realpath(os.path.join(lib_path, name))
        for name in os.listdir(lib_path)
        if name.startswith("libgrass_")
        and (name.endswith((".so", ".dylib")) or ".so." in name)
    }

    def load_order(library):
        name = os.path.basename(library).removeprefix("libgrass_").partition(".")[0]
        if name in _BASE_LIBRARY_NAMES:
            return (_BASE_LIBRARY_NAMES.index(name), name)
        return (len(_BASE_LIBRARY_NAMES), name)

    remaining = sorted(libraries, key=load_order)
    while remaining:
        failed = []
        errors = []
        for library in remaining:
            try:
                ctypes.CDLL(library)
            except OSError as error:
                failed.append(library)
                errors.append((library, error))
        if len(failed) == len(remaining):
            return errors
        remaining = failed
    return []


def _load_libraries(lib_path):
    """Load the GRASS libraries unless they already find each other.

    Loading the libraries into the process by their full paths makes them
    find each other even when the process was started without the library
    search path variable and the search path recorded in the libraries at
    build time does not point to them, e.g., for an installation moved
    away from the prefix it was configured with.
    """
    core_libraries = _core_libraries(lib_path)
    if not core_libraries:
        return
    try:
        # The core library loads by its full path when the GRASS libraries it
        # depends on resolve through the search path recorded at build time,
        # the search path variable, or a registration with the linker, and
        # then all the libraries find each other without any loading here.
        ctypes.CDLL(core_libraries[0])
    except OSError:
        pass
    else:
        return
    for library, error in _preload_libraries(lib_path):
        if os.path.basename(library).startswith("libgrass_gis."):
            msg = f"The GRASS core library failed to load ({library}: {error})"
            raise ImportError(msg)


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
    _load_libraries(_lib_path)

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
