"""Minimal analytical tests for r.sim.water (SIMWE).

These tests use small, hand-crafted domains where the expected output can be
reasoned about from first principles rather than from pre-computed reference
values.  Regression tests against realistic data (nc_spm dataset) live in
testsuite/test_r_sim_water.py.

The two physical principles underlying each test are:

Mass conservation:
    No rainfall input -> no runoff output.  Depth must be identically zero.

Kinematic wave approximation with Manning's equation (steady state):
    On a uniform slope S with rainfall rate R and Manning's roughness n:

        discharge per unit width:  q(x) = R * x
        water depth:               h(x) = (n * q(x) / S^0.5)^(3/5)

    This gives the following directional relationships that each test
    verifies for the aggregate (summed) depth over the domain:

        more rainfall R    -> greater depth   (h proportional to R^(3/5))
        higher Manning n   -> greater depth   (h proportional to n^(3/5))
        steeper slope S    -> shallower depth (h proportional to S^(-3/10))
        more infiltration  -> shallower depth (net R reduced)
        further downstream -> greater depth   (q, and thus h, grows with x)

All tests use:
  - a 1-row strip so flow is strictly one-dimensional and easy to reason about
  - nprocs=1 for determinism
  - a fixed random_seed for reproducibility
  - explicit nwalkers well above the default (2 x cells) to keep Monte Carlo
    noise low enough for the aggregate comparisons to be reliable
"""

import os

import numpy as np

import grass.script as gs
from grass.tools import Tools

# Simulation constants shared across tests.  Chosen so that:
#   - results are deterministic and reproducible
#   - statistical noise is negligible for the aggregate comparisons used
#   - individual tests finish in well under a second each
SEED = 42
NWALKERS = 10000  # 2000/cell for a 1x5 domain; default would be 10
NITERATIONS = 10  # minutes; default, long enough for near-steady state
NPROCS = 1  # single thread, fully deterministic
RAIN = 100  # mm/hr; generous signal-to-noise on a small domain


def _run_sim(session, *, rain_value=RAIN, infil_value=0, man_value=0.1):
    """Run r.sim.water on the session's terrain; return depth as ndarray.

    Assumes the session already contains rasters named elevation, dx, and dy
    (as created by the east_slope_session fixture or inline setup).
    """
    tools = Tools(session=session)
    result = tools.r_sim_water(
        elevation="elevation",
        dx="dx",
        dy="dy",
        depth=np.array,
        rain_value=rain_value,
        infil_value=infil_value,
        man_value=man_value,
        nwalkers=NWALKERS,
        niterations=NITERATIONS,
        random_seed=SEED,
        nprocs=NPROCS,
    )
    return np.asarray(result)


def test_no_rain_produces_no_depth(east_slope_session):
    """Zero rainfall must yield zero water depth everywhere.

    With no water input there can be no runoff.  This holds exactly
    regardless of seed or number of walkers: no particles are created.
    """
    depth = _run_sim(east_slope_session, rain_value=0)
    assert np.all(depth == 0), f"Expected all-zero depth with zero rain:\n{depth}"


def test_rain_produces_positive_depth(east_slope_session):
    """Rainfall on a slope must create positive water depth somewhere."""
    depth = _run_sim(east_slope_session)
    assert np.sum(depth) > 0, "Expected positive total depth with rainfall on a slope"


def test_more_rain_gives_more_depth(east_slope_session):
    """Doubling rainfall must increase total water depth.

    From Manning's kinematic wave: h proportional to R^(3/5), so doubling R
    increases depth by approximately 52 percent.  Only the direction (greater)
    is checked.
    """
    depth_low = _run_sim(east_slope_session, rain_value=50)
    depth_high = _run_sim(east_slope_session, rain_value=100)
    assert np.sum(depth_high) > np.sum(depth_low), (
        f"Total depth should increase with rainfall rate: "
        f"sum(rain=50)={np.sum(depth_low):.3e}, "
        f"sum(rain=100)={np.sum(depth_high):.3e}"
    )


def test_higher_manning_gives_more_depth(east_slope_session):
    """Rougher surface (higher Manning's n) must increase water depth.

    From Manning's kinematic wave: h proportional to n^(3/5).  A 10x increase
    in n raises depth by approximately 4x.  Only the direction is checked.
    """
    depth_smooth = _run_sim(east_slope_session, man_value=0.05)
    depth_rough = _run_sim(east_slope_session, man_value=0.5)
    assert np.sum(depth_rough) > np.sum(depth_smooth), (
        f"Total depth should increase with Manning's n: "
        f"sum(n=0.05)={np.sum(depth_smooth):.3e}, "
        f"sum(n=0.5)={np.sum(depth_rough):.3e}"
    )


def test_infiltration_reduces_depth(east_slope_session):
    """Adding overland-flow infiltration must reduce water depth.

    Infiltration removes water from the flowing sheet, decreasing depth
    and discharge.
    """
    depth_no_infil = _run_sim(east_slope_session, rain_value=RAIN, infil_value=0)
    depth_with_infil = _run_sim(
        east_slope_session, rain_value=RAIN, infil_value=RAIN // 2
    )
    assert np.sum(depth_no_infil) > np.sum(depth_with_infil), (
        f"Total depth should decrease when infiltration is added: "
        f"sum(no infil)={np.sum(depth_no_infil):.3e}, "
        f"sum(with infil)={np.sum(depth_with_infil):.3e}"
    )


def test_depth_increases_downstream(tmp_path):
    """Water must be deeper in the downslope half of a uniform slope.

    Continuity: q(x) = R * x grows linearly with distance x from the
    divide.  Manning: h(x) = (n * q / S^0.5)^(3/5) proportional to x^(3/5),
    also increases downstream.

    Domain: 1 row x 6 columns, split into upslope cols 0-2 and
    downslope cols 3-5.  Total depth in the downslope half must exceed
    that in the upslope half.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=6, s=0, n=1, res=1)
        # elevation = 7 - col() gives [6, 5, 4, 3, 2, 1] west to east
        tools.r_mapcalc(expression="elevation = 7 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        depth = _run_sim(session)
        flat = depth.flatten()  # shape (6,) for a 1-row raster

        upslope_sum = float(np.sum(flat[:3]))
        downslope_sum = float(np.sum(flat[3:]))
        assert downslope_sum > upslope_sum, (
            f"Downslope total depth ({downslope_sum:.3e}) should exceed "
            f"upslope total depth ({upslope_sum:.3e})"
        )


def test_steeper_slope_gives_less_depth(tmp_path):
    """A steeper slope must produce shallower water depth for the same rainfall.

    From Manning's kinematic wave: h proportional to S^(-3/10).  Doubling the
    slope magnitude S reduces depth by approximately 19 percent.  Only the
    direction is checked.

    Two sessions are created with matching elevation rasters so that the
    dx values are physically consistent with the terrain:
      gentle: elevation = 6 - col()      (drops 1 m per cell), dx = 1
      steep:  elevation = 11 - 2*col()   (drops 2 m per cell), dx = 2
    """

    def setup_and_run(sub_path, ele_expr, dx_value):
        project = sub_path / "simwe"
        gs.create_project(project)
        with gs.setup.init(project, env=os.environ.copy()) as session:
            tools = Tools(session=session)
            tools.g_region(w=0, e=5, s=0, n=1, res=1)
            tools.r_mapcalc(expression=f"elevation = {ele_expr}")
            tools.r_mapcalc(expression=f"dx = {dx_value}")
            tools.r_mapcalc(expression="dy = 0.0")
            return np.asarray(_run_sim(session))

    depth_gentle = setup_and_run(tmp_path / "gentle", "6 - col()", 1)
    depth_steep = setup_and_run(tmp_path / "steep", "11 - 2 * col()", 2)

    assert np.sum(depth_gentle) > np.sum(depth_steep), (
        f"Gentler slope depth ({np.sum(depth_gentle):.3e}) should exceed "
        f"steeper slope depth ({np.sum(depth_steep):.3e})"
    )


def test_discharge_positive_with_rain(east_slope_session):
    """Rainfall on a slope must produce positive water discharge."""
    tools = Tools(session=east_slope_session)
    discharge = np.asarray(
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            discharge=np.array,
            rain_value=RAIN,
            nwalkers=NWALKERS,
            niterations=NITERATIONS,
            random_seed=SEED,
            nprocs=NPROCS,
        )
    )
    assert np.sum(discharge) > 0, "Expected positive discharge with rainfall on a slope"
