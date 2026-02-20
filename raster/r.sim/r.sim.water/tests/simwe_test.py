import os

import numpy as np
import pytest

import grass.script as gs
from grass.tools import Tools


def test_slope(tmp_path, session):
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=3, s=0, n=2, res=1)
        elevation = np.array([[0, 0, 0], [0, 0, 0]])
        dx, dy = tools.r_slope_aspect(elevation=elevation, dx=np.array, dy=np.array)
        depth = tools.r_sim_water(
            elevation=elevation, dx=dx, dy=dy, depth=np.array, random_seed=42, nprocs=1
        )
        assert np.all(depth == [[0, 0, 0], [0, 0, 0]])


def test_slope2(tmp_path):
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=3, s=0, n=2, res=1)
        elevation = np.array([[3, 2, 1], [3, 2, 1]])
        dx, dy = tools.r_slope_aspect(elevation=elevation, dx=np.array, dy=np.array)
        depth = tools.r_sim_water(
            elevation=elevation,
            dx=dx,
            dy=dy,
            depth=np.array,
            nwalk=10000,
            rain_value=100000,
            niter=5,
            random_seed=42,
            nprocs=1,
        )
        assert np.all(depth == [[0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize("rows", range(3, 5))
@pytest.mark.parametrize("columns", range(3, 5))
def test_minimal_size_workflow(tmp_path, rows, columns):
    """Test whole workflow from elevation with smallest possible raster size"""
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=columns, s=0, n=rows, res=1)
        tools.r_mapcalc(expression="elevation = col()")
        tools.r_slope_aspect(elevation="elevation", dx="dx", dy="dy")
        # FIXME: check slope and aspect separately
        depth = tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth=np.array,
            rain_value=10,
            niter=5,
            random_seed=42,
            nprocs=1,
        )
        assert np.any(np.array(depth) > 0)

def test_minimal_size_workflow3x3(tmp_path):
    """Test whole workflow from elevation with smallest possible raster size"""
    rows = columns = 3
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=columns, s=0, n=rows, res=1)
        tools.r_mapcalc(expression="elevation = col()")
        tools.r_slope_aspect(elevation="elevation", dx="dx", dy="dy")
        # FIXME: check slope and aspect separately
        depth = tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth=np.array,
            rain_value=10,
            niter=5,
            random_seed=42,
            nprocs=1,
        )
        assert np.any(np.array(depth) > 0)
        # TODO: THis should not all be zero.
        assert np.any(
            np.array(depth)
            == pytest.approx(
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                abs=1e-3,
            )
        )


def test_depth(tmp_path):
    """Test small workflow for expected values

    Regression test with values obtained from the tool itself.
    """
    rows = 10
    columns = 5
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=columns, s=0, n=rows, res=1)
        elevation = tools.r_mapcalc_simple(
            expression=f"elevation = {columns} - col() + 1", output=np.array
        )
        assert np.all(np.array(elevation[2]) == [5, 4, 3, 2, 1])
        dx, dy = tools.r_slope_aspect(elevation=elevation, dx=np.array, dy=np.array)
        assert np.all(np.array(dx[4]) == [0, 1, 1, 1, 0])
        result = Tools(session=session, consistent_return_value=True).r_sim_water(
            elevation=elevation,
            dx=dx,
            dy=dy,
            depth=np.array,
            rain_value=10,
            niter=5,
            random_seed=42,
            man_value=0.5,
            nprocs=1,
        )
        print(result.stderr)
        #assert not result.stderr
        
        depth = result.arrays.depth
        assert np.any(np.array(depth) > 0)
        assert np.any(
            np.array(depth[4])
            == pytest.approx(
                np.array([0.0002977, 0.00032908, 0.00022698, 0.00067456, 0.00143326]),
                abs=1e-6,
            )
        )


def test_depth_direct(tmp_path):
    """Manually crafted inputs"""
    rows = 1
    columns = 5
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=columns, s=0, n=rows, res=1)
        elevation = np.array([5, 4, 3, 2, 1])
        dx = np.array([1, 1, 1, 1, 1])
        dy = np.array([0, 0, 0, 0, 0])
        depth = tools.r_sim_water(
            elevation=elevation,
            dx=dx,
            dy=dy,
            depth=np.array,
            rain_value=10,
            niter=5,
            random_seed=42,
            man_value=0.5,
            nprocs=1,
        )
        assert np.any(np.array(depth) > 0)
        assert np.any(
            np.array(depth)
            == pytest.approx(
                np.array(
                    [[0.00016991, 0.00019165, 0.00026603, 0.00035686, 0.00038382]]
                ),
                abs=1e-6,
            )
        )


def test_depth_same_rain_and_infiltration(tmp_path):
    """Manually crafted inputs"""
    rows = 1
    columns = 5
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=columns, s=0, n=rows, res=1)
        elevation = np.array([5, 4, 3, 2, 1])
        dx = np.array([1, 1, 1, 1, 1])
        dy = np.array([0, 0, 0, 0, 0])
        depth = tools.r_sim_water(
            elevation=elevation,
            dx=dx,
            dy=dy,
            depth=np.array,
            rain_value=1000,
            infil_value=10,
            niter=5,
            random_seed=42,
            man_value=0.5,
            nprocs=1,
        )
        assert np.any(np.array(depth) > 0)
        assert np.any(
            np.array(depth)
            == pytest.approx(
                np.array(
                    [[0.00250920, 0.00165546, 0.00485075, 0.00211141, 0.00165546]]
                ),
                abs=1e-6,
            )
        )


# FIXME: using depth array directly causes "'array' object has no attribute 'typecode'"
