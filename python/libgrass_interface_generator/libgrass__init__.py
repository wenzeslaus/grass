import ctypes
import os
import sys

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


_SESSION_VARIABLE_NAMES = ("GISRC", "GISBASE", "GIS_LOCK")
_environment_shared = None
_copied_names = set()


def _core_library():
    """Get the loaded core library with the environment functions set up"""
    core_libraries = _core_libraries(_lib_path) if _lib_path else []
    if not core_libraries:
        return None
    library = ctypes.CDLL(core_libraries[0])
    library.G_getenv_nofatal.restype = ctypes.c_char_p
    library.G_getenv_nofatal.argtypes = [ctypes.c_char_p]
    library.G_putenv.restype = None
    library.G_putenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    return library


def _is_environment_shared(library):
    """Check whether the C libraries see variables set in this process.

    GISBASE is the variable which libgis reads directly from the environment
    of its C runtime, so it is set to a probe value and read back.
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


def sync_environment(env=None):
    """Copy the session variables into the environment of the C libraries.

    On Windows, the C libraries may keep their own copy of the environment,
    so variables set in the running process are not visible to them unless
    copied. The copy is done when grass.lib is imported and by
    grass.script.setup.init. Call this function after changing the
    variables in any other way, e.g., after ending a session or setting a
    GRASS_ variable, so that the C libraries see the change, including
    removed variables.

    With *env*, the variables are copied from that environment instead,
    e.g., to use grass.lib with a session from init(..., env=env). The C
    libraries have one environment per process, so this session then also
    applies to processes started without an explicit environment, while
    os.environ is not changed. Use it when the process works with only one
    session, and call it again after ending the session. The C libraries
    keep the first session they read, so switching to another session in
    the same process is not supported on any platform.

    Copied are GISRC, GISBASE, GIS_LOCK, and all variables starting with
    GRASS_. Without *env*, the function does nothing outside of Windows and
    when the environment turns out to be shared.
    """
    global _environment_shared
    if env is None:
        env = os.environ
    windows = sys.platform.startswith("win")
    if env is os.environ and not windows:
        return
    library = _core_library()
    if library is None:
        return
    if env is os.environ:
        if _environment_shared is None:
            _environment_shared = _is_environment_shared(library)
        if _environment_shared:
            return
    names = {
        name
        for name in env
        if name in _SESSION_VARIABLE_NAMES or name.startswith("GRASS_")
    }
    for name in names:
        _set_c_variable(library, name, env[name], windows=windows)
    for name in _copied_names - names:
        _set_c_variable(library, name, None, windows=windows)
    _copied_names.clear()
    _copied_names.update(names)


def _set_c_variable(library, name, value, *, windows):
    """Set or, with value None, remove a variable as seen by the C libraries"""
    if windows:
        # The C runtime of the libraries uses the ANSI code page, and an empty
        # value removes the variable.
        library.G_putenv(name.encode("mbcs"), (value or "").encode("mbcs"))
    elif value is None:
        # The C libraries share the environment of the process with Python,
        # and these change it without changing os.environ.
        os.unsetenv(name)
    else:
        os.putenv(name, value)


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
