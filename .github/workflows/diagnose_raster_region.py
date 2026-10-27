"""Temporary diagnostics for test_raster_region.py timing out on Windows.

Usage within a session:

    python -u diagnose_raster_region.py environment
    python -u diagnose_raster_region.py test <test file> <timeout in seconds>
"""

import ctypes
import faulthandler
import os
import runpy
import sys
import threading
import time
from ctypes import wintypes

NAMES = ("GISRC", "GISBASE", "GIS_LOCK", "WIND_OVERRIDE", "GRASS_REGION")


def c_getenv_function(library_name):
    library = ctypes.CDLL(library_name)
    library.getenv.restype = ctypes.c_char_p
    library.getenv.argtypes = [ctypes.c_char_p]
    return library.getenv


# GRASS is built with MSYS2 MINGW64 (msvcrt.dll), Python uses the UCRT.
RUNTIMES = {name: c_getenv_function(name) for name in ("msvcrt", "ucrtbase")}


def show_environment(label):
    print(f"== Environment {label}")
    names = sorted(
        set(NAMES) | {name for name in os.environ if name.startswith("GRASS_")}
    )
    for name in names:
        values = {"os.environ": os.environ.get(name)}
        for runtime, getenv in RUNTIMES.items():
            value = getenv(name.encode("mbcs"))
            values[runtime] = value.decode("mbcs") if value is not None else None
        same = "" if values["os.environ"] == values["msvcrt"] else "  DIFFERENT"
        print(f"{name}: {values}{same}")


def show_region(label, region):
    print(
        f"{label}: rows={region.rows} cols={region.cols}"
        f" n={region.north} s={region.south} e={region.east} w={region.west}"
        f" nsres={region.nsres} ewres={region.ewres}"
    )


def follow_test_resampling_to_numpy(read):
    """Do the steps of test_resampling_to_numpy, optionally reading the raster"""
    import grass.lib.raster as libraster
    from grass.pygrass.gis.region import Region
    from grass.pygrass.raster import raster2numpy

    region = Region()
    show_region("Region()", region)
    for resolution, expected_rows in ((0.1, 400), (1, 40), (5, 8)):
        region.ewres = resolution
        region.nsres = resolution
        region.adjust()
        region.set_raster_region()
        show_region(f"  res={resolution} after set_raster_region", region)
        print(
            f"  Rast_window_rows={libraster.Rast_window_rows()}"
            f" Rast_window_cols={libraster.Rast_window_cols()}"
            f" (test expects {expected_rows} rows)"
        )
        if read:
            print(f"  len(raster2numpy)={len(raster2numpy(NAME))}")


NAME = "diagnose_raster_region"


def diagnose_environment():
    show_environment("before importing grass.lib")
    import grass.lib
    import grass.lib.gis as libgis
    import grass.script as gs

    show_environment("after importing grass.lib")
    print(f"_environment_shared={grass.lib._environment_shared}")
    print(f"_copied_names={sorted(grass.lib._copied_names)}")

    # Same as setUpClass of the test.
    gs.use_temp_region()
    gs.run_command("g.region", n=40, s=0, e=40, w=0, res=10)
    gs.mapcalc(f"{NAME} = row() + (10.0 * col())", overwrite=True)
    show_environment("after use_temp_region")
    print("g.region -p in a subprocess:")
    print(gs.read_command("g.region", flags="g"))

    print("== Without copying WIND_OVERRIDE (raster not read)")
    follow_test_resampling_to_numpy(read=False)

    print("== After copying WIND_OVERRIDE with G_putenv")
    libgis.G_putenv(b"WIND_OVERRIDE", os.environ["WIND_OVERRIDE"].encode("mbcs"))
    show_environment("after G_putenv")
    follow_test_resampling_to_numpy(read=True)


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def report_memory(start):
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    while True:
        time.sleep(15)
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        print(
            f"[{time.monotonic() - start:.0f} s]"
            f" working set {counters.WorkingSetSize / 2**20:.0f} MiB,"
            f" committed {counters.PagefileUsage / 2**20:.0f} MiB",
            file=sys.stderr,
            flush=True,
        )


def run_test(path, timeout):
    start = time.monotonic()
    threading.Thread(target=report_memory, args=(start,), daemon=True).start()
    faulthandler.dump_traceback_later(timeout, exit=True)
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        print(f"Finished after {time.monotonic() - start:.1f} s")


if __name__ == "__main__":
    if sys.argv[1] == "environment":
        diagnose_environment()
    else:
        run_test(sys.argv[2], int(sys.argv[3]))
