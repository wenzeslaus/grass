"""Temporary diagnostics: does pygrass see variables set after init?

Each scenario runs in a fresh process without a session in its environment,
starts a session with init (global environment), sets something afterwards
the way a user or a library function would, and records what pygrass (C
libraries through ctypes) and a subprocess (tool) see.

Usage:

    python -u diagnose_pygrass_environment.py [scenario...]
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SESSION_VARIABLES = (
    "GISRC",
    "GIS_LOCK",
    "WIND_OVERRIDE",
    "GRASS_REGION",
    "GRASS_MASK",
    "GRASS_VERBOSE",
    "GRASS_COMPRESSOR",
)


def start(path, name="xy"):
    """Create a project with a 4x4 region and a raster, start a session"""
    import grass.script as gs

    gs.create_project(path / name)
    session = gs.setup.init(path / name)
    gs.run_command("g.region", n=40, s=0, e=40, w=0, res=10)
    gs.mapcalc("m = row() + 10 * col()")
    return session


def region_seen():
    """Region as read by the C library (pygrass) and by a tool"""
    import grass.script as gs
    from grass.pygrass.gis.region import Region

    region = Region()
    tool = gs.parse_command("g.region", flags="g")
    return {
        "pygrass_rows_cols": [region.rows, region.cols],
        "tool_rows_cols": [int(tool["rows"]), int(tool["cols"])],
    }


def read_rows(name="m"):
    from grass.pygrass.raster import RasterRow

    with RasterRow(name) as raster:
        return [row.tolist() for row in raster]


def wind_file_rows_cols():
    """Region stored in the WIND file of the current mapset"""
    import grass.script as gs

    env = os.environ.copy()
    for name in ("WIND_OVERRIDE", "GRASS_REGION"):
        env.pop(name, None)
    tool = gs.parse_command("g.region", flags="g", env=env)
    return [int(tool["rows"]), int(tool["cols"])]


def scenario_init_read(path):
    start(path)
    return {"rows": read_rows(), **region_seen()}


def scenario_import_pygrass_before_init(path):
    from grass.pygrass.raster import RasterRow  # noqa: F401

    start(path)
    return {"rows": read_rows(), **region_seen()}


def scenario_use_temp_region(path):
    import grass.script as gs

    start(path)
    gs.use_temp_region()
    gs.run_command("g.region", n=20, s=0, e=20, w=0, res=10)
    return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_use_temp_region_after_first_read(path):
    import grass.script as gs

    start(path)
    before = read_rows()
    gs.use_temp_region()
    gs.run_command("g.region", n=20, s=0, e=20, w=0, res=10)
    return {
        "expected_rows_cols": [2, 2],
        "rows_before": before,
        "rows": read_rows(),
        **region_seen(),
    }


def scenario_region_manager(path):
    import grass.script as gs

    start(path)
    with gs.RegionManager(n=20, s=0, e=20, w=0, res=10):
        return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_region_manager_pygrass_write(path):
    """Write the region with pygrass inside RegionManager"""
    import grass.script as gs
    from grass.pygrass.gis.region import Region

    start(path)
    with gs.RegionManager(n=40, s=0, e=40, w=0, res=10):
        region = Region()
        region.north = 10
        region.south = 0
        region.east = 10
        region.west = 0
        region.adjust()
        region.write()
        inside = region_seen()
    return {
        "expected_wind_after": [4, 4],
        "inside": inside,
        "wind_after": wind_file_rows_cols(),
    }


def scenario_use_temp_region_pygrass_write(path):
    """Write the region with pygrass after use_temp_region (v.rast.bufferstats)"""
    import grass.script as gs
    from grass.pygrass.gis.region import Region

    start(path)
    gs.use_temp_region()
    region = Region()
    region.north = 10
    region.south = 0
    region.east = 10
    region.west = 0
    region.adjust()
    region.write()
    inside = region_seen()
    gs.del_temp_region()
    return {
        "expected_wind_after": [4, 4],
        "inside": inside,
        "wind_after": wind_file_rows_cols(),
    }


def scenario_region_manager_env(path):
    import grass.script as gs

    start(path)
    with gs.RegionManagerEnv(n=20, s=0, e=20, w=0, res=10):
        return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_grass_region_variable(path):
    import grass.script as gs

    start(path)
    os.environ["GRASS_REGION"] = gs.region_env(n=20, s=0, e=20, w=0, res=10)
    return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_late_import_grass_region_variable(path):
    """Import pygrass only after setting the variable"""
    import grass.script as gs

    start(path)
    os.environ["GRASS_REGION"] = gs.region_env(n=20, s=0, e=20, w=0, res=10)
    return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_grass_region_variable_with_sync(path):
    import grass.lib
    import grass.script as gs

    start(path)
    os.environ["GRASS_REGION"] = gs.region_env(n=20, s=0, e=20, w=0, res=10)
    grass.lib.sync_environment()
    return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_use_temp_region_with_sync(path):
    """WIND_OVERRIDE is not among the copied variables"""
    import grass.lib
    import grass.script as gs

    start(path)
    gs.use_temp_region()
    gs.run_command("g.region", n=20, s=0, e=20, w=0, res=10)
    grass.lib.sync_environment()
    return {"expected_rows_cols": [2, 2], "rows": read_rows(), **region_seen()}


def scenario_mask_manager_with_sync(path):
    import grass.lib
    import grass.script as gs

    start(path)
    gs.mapcalc("mask_source = if(row() == 1, 1, null())")
    with gs.MaskManager():
        gs.run_command("r.mask", raster="mask_source")
        grass.lib.sync_environment()
        return {
            "expected": "only the first row has values",
            "rows": read_rows(),
            "tool_univar_n": int(gs.parse_command("r.univar", map="m", flags="g")["n"]),
        }


def scenario_mask_manager(path):
    import grass.script as gs

    start(path)
    gs.mapcalc("mask_source = if(row() == 1, 1, null())")
    with gs.MaskManager():
        gs.run_command("r.mask", raster="mask_source")
        return {
            "expected": "only the first row has values",
            "rows": read_rows(),
            "tool_univar_n": int(gs.parse_command("r.univar", map="m", flags="g")["n"]),
        }


def scenario_r_mask_without_manager(path):
    """Mask stored in the mapset, no variable involved (control)"""
    import grass.script as gs

    start(path)
    gs.mapcalc("mask_source = if(row() == 1, 1, null())")
    gs.run_command("r.mask", raster="mask_source")
    return {
        "expected": "only the first row has values",
        "rows": read_rows(),
        "tool_univar_n": int(gs.parse_command("r.univar", map="m", flags="g")["n"]),
    }


def scenario_verbose_after_init(path):
    import grass.lib.gis as libgis

    start(path)
    os.environ["GRASS_VERBOSE"] = "0"
    return {"expected": 0, "G_verbose": libgis.G_verbose()}


def scenario_compressor_after_init(path):
    import numpy as np

    import grass.script as gs
    from grass.pygrass.raster import numpy2raster

    start(path)
    os.environ["GRASS_COMPRESSOR"] = "LZ4"
    numpy2raster(np.ones((4, 4)), "FCELL", "written", overwrite=True)
    info = gs.read_command("r.compress", flags="g", map="written")
    return {"expected": "LZ4", "r.compress -g": info.strip()}


def scenario_switch_session(path):
    import grass.script as gs

    session = start(path, "first")
    first = read_rows()
    session.finish()
    gs.create_project(path / "second")
    with gs.setup.init(path / "second"):
        gs.run_command("g.region", n=20, s=0, e=20, w=0, res=10)
        gs.mapcalc("m = 100")
        return {
            "expected": "second: 2x2 of 100",
            "first": first,
            "second": read_rows(),
            "pygrass_project": libgis_value("G_location"),
        }


def scenario_switch_mapset_with_tool(path):
    import grass.script as gs

    start(path)
    first = libgis_value("G_mapset")
    gs.run_command("g.mapset", mapset="other", flags="c")
    return {
        "expected": "other",
        "before": first,
        "pygrass_mapset": libgis_value("G_mapset"),
        "tool_mapset": gs.gisenv()["MAPSET"],
    }


def scenario_custom_env_with_sync(path):
    import grass.lib
    import grass.script as gs

    gs.create_project(path / "xy")
    with gs.setup.init(path / "xy", env=os.environ.copy()) as session:
        gs.run_command("g.region", n=40, s=0, e=40, w=0, res=10, env=session.env)
        gs.mapcalc("m = row() + 10 * col()", env=session.env)
        grass.lib.sync_environment(session.env)
        return {"rows": read_rows(), "GISRC_in_os_environ": "GISRC" in os.environ}


def scenario_vector(path):
    from grass.pygrass.vector import VectorTopo
    from grass.pygrass.vector.geometry import Point

    start(path)
    with VectorTopo("points", mode="w") as vector:
        vector.write(Point(1, 1))
        vector.write(Point(2, 2))
    with VectorTopo("points") as vector:
        return {"expected": 2, "points": vector.number_of("points")}


def scenario_temporal_c_interface(path):
    """grass.temporal calls the C libraries in a separate process"""
    import grass.temporal as tgis

    start(path)
    tgis.init()
    interface = tgis.get_tgis_c_library_interface()
    info = interface.read_raster_info("m", "PERMANENT")
    return {
        "expected": "PERMANENT, 4x4",
        "mapset": interface.get_mapset(),
        "rows_cols": [info["rows"], info["cols"]] if info else None,
    }


def libgis_value(function_name):
    import grass.lib.gis as libgis
    from grass.script.utils import decode

    return decode(getattr(libgis, function_name)())


SCENARIOS = {
    name.removeprefix("scenario_"): function
    for name, function in globals().items()
    if name.startswith("scenario_")
}


def run_scenario(name):
    if not name.startswith("late_import_"):
        # Import before the session starts as a script would at its top.
        # The import copies the environment, so importing later would copy
        # the variables set before the import.
        import grass.pygrass.vector  # noqa: F401
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        result = SCENARIOS[name](Path(tmp))
        print("RESULT " + json.dumps(result))
        # Leave the session files to the operating system on failures.
        import grass.script as gs

        if "GISRC" in os.environ:
            gs.setup.finish()


def main(names):
    env = os.environ.copy()
    for variable in SESSION_VARIABLES:
        env.pop(variable, None)
    for name in names or SCENARIOS:
        start_time = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, "-u", __file__, "--scenario", name],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"== {name}: TIMEOUT")
            continue
        duration = time.monotonic() - start_time
        results = [
            line.removeprefix("RESULT ")
            for line in process.stdout.splitlines()
            if line.startswith("RESULT ")
        ]
        print(f"== {name} (exit code {process.returncode}, {duration:.1f} s)")
        if results:
            for key, value in json.loads(results[-1]).items():
                print(f"  {key}: {json.dumps(value)}")
        else:
            print("  no result; end of stderr:")
            for line in process.stderr.strip().splitlines()[-12:]:
                print(f"    {line}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--scenario":
        run_scenario(sys.argv[2])
    else:
        main(sys.argv[1:])
