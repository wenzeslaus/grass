import ctypes
import os
import sys

from .ctypes_loader import add_library_search_dirs


def _core_library_path(lib_path):
    """Get the path of the core shared library in a directory, or None"""
    try:
        names = sorted(os.listdir(lib_path))
    except OSError:
        return None
    for name in names:
        if name.startswith(("libgrass_gis.", "grass_gis.")) and (
            name.endswith((".so", ".dylib", ".dll")) or ".so." in name
        ):
            return os.path.join(lib_path, name)
    return None


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
    # Four levels up from grass/lib (through lib, grass, python, and etc) is
    # the installation in the traditional layout, and (through lib, grass,
    # site-packages, and python3.X) the library directory in the FHS layout.
    for _ in range(4):
        path = os.path.dirname(path)
    candidates = [os.path.join(path, "lib"), path]
    gisbase = os.environ.get("GISBASE")
    if gisbase:
        candidates.append(os.path.join(gisbase, "lib"))
    for candidate in candidates:
        if _core_library_path(candidate):
            return candidate
    return None


# Most GRASS libraries need libgrass_gis and libgrass_gis needs
# libgrass_datetime, so these two are loaded first. A library loaded before
# the GRASS libraries it needs makes the dynamic linker find them through the
# process-startup search path, which may point to another GRASS installation.
# A second copy of libgrass_gis breaks the libraries using it because they
# rely on the session state it holds.
_BASE_LIBRARY_NAMES = ("datetime", "gis")


def _preload_libraries(lib_path):
    """Load the GRASS libraries in a directory into the process by full path.

    Libraries can be loaded only after the libraries they depend on, so
    loading is repeated until it makes no progress. Windows DLLs are not
    matched because Windows finds dependent DLLs without this loading.

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


def _load_libraries_if_needed(lib_path):
    """Load the GRASS libraries unless they already find each other.

    Loading the libraries into the process by their full paths makes them
    find each other even when the process was started without the library
    search path variable and the search path recorded in the libraries at
    build time does not point to them, e.g., for an installation moved
    away from the prefix it was configured with.
    """
    core_library = _core_library_path(lib_path)
    if not core_library:
        return
    try:
        # Nothing more is needed when the core library finds the GRASS
        # libraries it depends on by itself.
        ctypes.CDLL(core_library)
    except OSError:
        pass
    else:
        return
    for library, error in _preload_libraries(lib_path):
        if os.path.basename(library).startswith("libgrass_gis."):
            msg = (
                f"Cannot load the GRASS core library {library}: {error}. Check"
                " that the GRASS installation and its dependencies are complete."
            )
            raise ImportError(msg)


_SESSION_VARIABLE_NAMES = ("GISRC", "GISBASE", "GIS_LOCK")
# G_putenv formats the name and the value into a buffer of this size.
_PUTENV_BUFFER_SIZE = 1024
_environment_shared = None
_copied_names = set()


def _libgis():
    """Get the loaded core library with the environment functions set up"""
    core_library = _core_library_path(_lib_path) if _lib_path else None
    if not core_library:
        return None
    library = ctypes.CDLL(core_library)
    library.G_getenv_nofatal.restype = ctypes.c_char_p
    library.G_getenv_nofatal.argtypes = [ctypes.c_char_p]
    library.G_putenv.restype = None
    library.G_putenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    return library


def _is_environment_shared(library):
    """Check whether the C libraries see variables set in this process.

    G_getenv_nofatal reads GISBASE directly from the environment of the C
    runtime of libgis, so GISBASE is set to a probe value and read back.
    """
    original = os.environ.get("GISBASE")
    probe = f"environment-probe-{os.getpid()}"
    os.environ["GISBASE"] = probe
    try:
        seen = library.G_getenv_nofatal(b"GISBASE")
    finally:
        if original is None:
            del os.environ["GISBASE"]
        else:
            os.environ["GISBASE"] = original
    return seen == probe.encode()


def _set_c_variable(library, name, value):
    """Set a variable in the C runtime of the libraries, or remove it with ""

    A value which the C runtime cannot take is skipped, as the libraries
    could not use it, and failing would break the grass.lib import.
    """
    try:
        # The C runtime of the libraries uses the ANSI code page.
        encoded_name = name.encode("mbcs")
        encoded_value = value.encode("mbcs")
    except UnicodeEncodeError:
        return
    if len(encoded_name) + len(encoded_value) + 2 > _PUTENV_BUFFER_SIZE:
        return
    library.G_putenv(encoded_name, encoded_value)


def sync_environment():
    """Copy the session variables into the environment of the C libraries.

    On Windows, the C libraries may keep their own copy of the environment,
    so variables set in the running process are not visible to them unless
    copied. The copy is done when grass.lib is imported and by
    grass.script.setup.init without a custom env. Call this function after
    changing the variables in any other way, e.g., after ending a session
    or setting a GRASS_ variable, so that the C libraries see the change.

    The function copies GISRC, GISBASE, GIS_LOCK, and all variables starting
    with GRASS_ from os.environ. The session variables and the variables
    copied before are removed when they are no longer set. A value outside
    of the ANSI code page or longer than about 1000 bytes is not copied.
    The function does nothing outside of Windows and when the environment
    turns out to be shared.

    The C libraries keep the first session they read, so switching to
    another session in the same process is not supported on any platform.
    """
    global _environment_shared
    if not sys.platform.startswith("win"):
        return
    library = _libgis()
    if library is None:
        return
    if _environment_shared is None:
        _environment_shared = _is_environment_shared(library)
    if _environment_shared:
        return
    names = {
        name
        for name in os.environ
        if name in _SESSION_VARIABLE_NAMES or name.startswith("GRASS_")
    }
    for name in names:
        _set_c_variable(library, name, os.environ[name])
    for name in (_copied_names | set(_SESSION_VARIABLE_NAMES)) - names:
        _set_c_variable(library, name, "")
    _copied_names.clear()
    _copied_names.update(names)


_lib_path = _find_lib_path()
if _lib_path:
    # Registering the directory makes the wrapper modules resolve the
    # libraries by absolute path, skipping the expensive
    # ctypes.util.find_library() fallback.
    add_library_search_dirs([_lib_path])
    _load_libraries_if_needed(_lib_path)
    sync_environment()

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
