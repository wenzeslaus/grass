import ctypes
import os
from pathlib import Path

from .ctypes_loader import add_library_search_dirs


def _find_install_path():
    """Find the GRASS installation this package belongs to.

    The package is installed at <install path>/etc/python/grass/lib, so the
    installation is derived from the location of this file, which also
    guarantees that the libraries match the wrapper modules generated for
    them. GISBASE is used as a fallback for other layouts.
    """
    path = Path(__file__).resolve().parents[4]
    if (path / "bin").is_dir() and (path / "lib").is_dir():
        return path
    gisbase = os.environ.get("GISBASE")
    if gisbase and Path(gisbase).is_dir():
        return Path(gisbase)
    return None


def _load_libraries(install_path):
    """Load the GRASS libraries unless the dynamic linker finds them itself.

    Loading the libraries into the process by their full paths makes them
    find each other even when the process was started without the library
    search path variable set, which is the case for a process which sets up
    a session only after it started (grass.script.setup.init).
    """
    gis_libraries = sorted(
        {
            library.resolve()
            for pattern in ("libgrass_gis.so*", "libgrass_gis.*.so", "libgrass_gis*.dylib")
            for library in (install_path / "lib").glob(pattern)
        }
    )
    if not gis_libraries:
        # Nothing to load, e.g., on Windows where the loader registers the
        # PATH directories with os.add_dll_directory instead.
        return
    try:
        # When the dynamic linker resolves the core library by its file name
        # alone, the libraries are on its search path, e.g., inside a session
        # or in an installation registered with the linker, and they will
        # find each other without any loading here.
        ctypes.CDLL(gis_libraries[0].name)
    except OSError:
        pass
    else:
        return
    # Lazy import to avoid the cost when no loading is needed.
    from grass.app.runtime import preload_dynamic_libraries

    failures = preload_dynamic_libraries(install_path=install_path)
    for library, error in failures:
        if library.name.startswith("libgrass_gis."):
            msg = f"The GRASS core library failed to load ({library}: {error})"
            raise ImportError(msg)


_install_path = _find_install_path()
if _install_path:
    # Registering the directory makes the wrapper modules resolve the
    # libraries by absolute path, skipping the expensive
    # ctypes.util.find_library() fallback.
    add_library_search_dirs([os.fspath(_install_path / "lib")])
    _load_libraries(_install_path)

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
