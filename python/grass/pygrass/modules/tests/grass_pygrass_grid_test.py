"""Test main functions of PyGRASS GridModule"""

import multiprocessing
import os

import pytest

import grass.script as gs
from grass.pygrass.modules.grid import GridModule


def max_processes():
    """Get max useful number of parallel processes to run"""
    return min(multiprocessing.cpu_count(), 4)


def run_grid_module(tool, grid_kwargs, run_kwargs, env):
    """Run GridModule in the given session environment.

    A module-level function is picklable, so it works as a process target
    under any start method, unlike a locally defined function. The session
    environment is passed explicitly because with the forkserver start
    method, a child process sees the environment from the time the first
    process was started, not the current one.
    """
    os.environ.update(env)
    grid = GridModule(tool, **grid_kwargs)
    grid.run(**run_kwargs)


# GridModule uses C libraries which can easily initialize only once
# and thus can't easily change location/mapset, so we use a subprocess
# to separate individual GridModule calls.
def run_in_subprocess(tool, grid_kwargs, env, run_kwargs=None, expect_failure=False):
    """Run GridModule with the given parameters in a separate process"""
    process = multiprocessing.Process(
        target=run_grid_module, args=(tool, grid_kwargs, run_kwargs or {}, env)
    )
    process.start()
    process.join()
    if expect_failure:
        assert process.exitcode != 0, "GridModule run succeeded unexpectedly"
    else:
        assert process.exitcode == 0, "GridModule run failed"


@pytest.mark.needs_solo_run
@pytest.mark.parametrize("processes", list(range(1, max_processes() + 1)) + [None])
def test_processes(tmp_path, processes):
    """Check that running with multiple processes works"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=50, w=0, e=50, res=1, env=session.env)

        surface = "surface"
        gs.run_command("r.surf.fractal", output=surface, env=session.env)

        run_in_subprocess(
            "r.slope.aspect",
            {
                "width": 10,
                "height": 5,
                "overlap": 2,
                "processes": processes,
                "elevation": surface,
                "slope": "slope",
                "aspect": "aspect",
            },
            session.env,
        )

        info = gs.raster_info("slope", env=session.env)
        assert info["min"] > 0


# @pytest.mark.parametrize("split", [False])  # True does not work.


@pytest.mark.parametrize("width", [5, 10, 50])  # None does not work.
@pytest.mark.parametrize("height", [5, 10, 50])
def test_tiling_schemes(tmp_path, width, height):
    """Check that different shapes of tiles work"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=50, w=0, e=50, res=1, env=session.env)

        surface = "surface"
        gs.run_command("r.surf.fractal", output=surface, env=session.env)

        run_in_subprocess(
            "r.slope.aspect",
            {
                "width": width,
                "height": height,
                "overlap": 2,
                "processes": max_processes(),
                "elevation": surface,
                "slope": "slope",
                "aspect": "aspect",
            },
            session.env,
        )

        info = gs.raster_info("slope", env=session.env)
        assert info["min"] > 0


@pytest.mark.parametrize("overlap", [0, 1, 2, 5])
def test_overlaps(tmp_path, overlap):
    """Check that overlap accepts different values"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=50, w=0, e=50, res=1, env=session.env)
        surface = "surface"
        gs.run_command("r.surf.fractal", output=surface, env=session.env)

        run_in_subprocess(
            "r.slope.aspect",
            {
                "width": 10,
                "height": 5,
                "overlap": overlap,
                "processes": max_processes(),
                "elevation": surface,
                "slope": "slope",
                "aspect": "aspect",
            },
            session.env,
        )

        info = gs.raster_info("slope", env=session.env)
        assert info["min"] > 0


@pytest.mark.parametrize("clean", [True, False])
@pytest.mark.parametrize("surface", ["surface", "non_exist_surface"])
def test_cleans(tmp_path, clean, surface):
    """Check that temporary mapsets are cleaned when appropriate"""
    project = tmp_path / "test"
    mapset_prefix = "abc"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=50, w=0, e=50, res=1, env=session.env)
        if surface == "surface":
            gs.run_command("r.surf.fractal", output=surface, env=session.env)

        run_in_subprocess(
            "r.slope.aspect",
            {
                "width": 10,
                "height": 5,
                "overlap": 0,
                "processes": max_processes(),
                "elevation": surface,
                "slope": "slope",
                "aspect": "aspect",
                "mapset_prefix": mapset_prefix,
            },
            session.env,
            run_kwargs={"clean": clean},
            expect_failure=surface != "surface",
        )

        prefixed = 0
        for item in project.iterdir():
            if item.is_dir():
                if clean:
                    # We know right away something is wrong.
                    assert not item.name.startswith(mapset_prefix), "Mapset not cleaned"
                else:
                    # We need to see if there is at least one prefixed mapset.
                    prefixed += int(item.name.startswith(mapset_prefix))
        if not clean:
            assert prefixed, "Not even one prefixed mapset"


@pytest.mark.parametrize("patch_backend", [None, "r.patch", "RasterRow"])
def test_patching_backend(tmp_path, patch_backend):
    """Check patching backend works"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=50, w=0, e=50, res=1, env=session.env)

        points = "points"
        reference = "reference"
        gs.run_command("v.random", output=points, npoints=100, env=session.env)
        gs.run_command(
            "v.to.rast",
            input=points,
            output=reference,
            type="point",
            use="cat",
            env=session.env,
        )

        run_in_subprocess(
            "v.to.rast",
            {
                "width": 10,
                "height": 5,
                "overlap": 0,
                "patch_backend": patch_backend,
                "processes": max_processes(),
                "input": points,
                "output": "output",
                "type": "point",
                "use": "cat",
            },
            session.env,
        )

        mean_ref = float(
            gs.parse_command("r.univar", map=reference, flags="g", env=session.env)[
                "mean"
            ]
        )
        mean = float(
            gs.parse_command("r.univar", map="output", flags="g", env=session.env)[
                "mean"
            ]
        )
        assert abs(mean - mean_ref) < 0.0001


@pytest.mark.parametrize(
    ("width", "height", "processes"),
    [
        (None, None, max_processes()),
        (10, None, max_processes()),
        (None, 5, max_processes()),
    ],
)
def test_tiling(tmp_path, width, height, processes):
    """Check auto adjusted tile size based on processes"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=50, w=0, e=50, res=1, env=session.env)

        surface = "surface"
        gs.run_command("r.surf.fractal", output=surface, env=session.env)

        run_in_subprocess(
            "r.slope.aspect",
            {
                "width": width,
                "height": height,
                "overlap": 2,
                "processes": processes,
                "elevation": surface,
                "slope": "slope",
                "aspect": "aspect",
            },
            session.env,
        )

        info = gs.raster_info("slope", env=session.env)
        assert info["min"] > 0


@pytest.mark.needs_solo_run
@pytest.mark.parametrize(
    ("processes", "backend"),
    [
        (1, "RasterRow"),
        (9, "RasterRow"),
        (9, "r.patch"),
        (10, "RasterRow"),
        (10, "r.patch"),
    ],
)
def test_patching_error(tmp_path, processes, backend):
    """Check auto adjusted tile size based on processes"""
    project = tmp_path / "test"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        gs.run_command("g.region", s=0, n=10, w=0, e=10, res=0.1, env=session.env)
        surface = "fractal"

        run_in_subprocess(
            "r.surf.fractal",
            {
                "overlap": 0,
                "processes": processes,
                "output": surface,
                "patch_backend": backend,
                "debug": True,
            },
            session.env,
        )

        info = gs.parse_command("r.univar", flags="g", map=surface, env=session.env)
        assert int(info["null_cells"]) == 0
