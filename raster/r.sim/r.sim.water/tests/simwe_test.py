"""Minimal analytical tests for r.sim.water (SIMWE).

Tests use small, hand-crafted domains where expected output can be reasoned
about from first principles rather than from pre-computed reference values.
"""

import os

import numpy as np
import pytest

import grass.script as gs
from grass.tools import Tools

# Fixed seed and single thread make results fully deterministic.
SEED = 42
NPROCS = 1

# 10000 walkers on a 1-row x 5-column domain gives 2000 walkers/cell.
# The default (2 x cells) would be 10, which is too noisy for aggregate
# comparisons. Unless otherwise noted, tests use this domain and walker count.
NWALKERS = 10000

NITERATIONS = 10  # minutes; long enough for near-steady state on a small domain
RAIN = 100  # mm/hr; generous signal-to-noise ratio on a small domain


def run_sim(session, *, random_seed=SEED, **kwargs):
    """Run r.sim.water on the session's terrain; return depth as ndarray.

    Assumes the session already contains rasters named elevation, dx, and dy
    (as created by a fixture or inline setup). Additional keyword arguments
    are passed directly to r.sim.water (e.g., rain_value, man_value, rain, man,
    nwalkers).
    """
    defaults = {
        "rain_value": RAIN,
        "infil_value": 0,
        "man_value": 0.1,
        "nwalkers": NWALKERS,
    }
    defaults.update(kwargs)
    # On the command line, an unwanted parameter is simply omitted, so we
    # simulate that.
    defaults = {k: v for k, v in defaults.items() if v is not None}
    tools = Tools(session=session)
    result = tools.r_sim_water(
        elevation="elevation",
        dx="dx",
        dy="dy",
        depth=np.array,
        niterations=NITERATIONS,
        random_seed=random_seed,
        nprocs=NPROCS,
        **defaults,
    )
    return np.asarray(result)


def test_no_rain_produces_no_depth(east_slope_session):
    """Zero rainfall must yield zero water depth everywhere.

    With no water input there can be no runoff. This holds exactly
    regardless of seed or number of walkers: no particles are created.
    """
    depth = run_sim(east_slope_session, rain_value=0)
    assert np.all(depth == 0), f"Expected all-zero depth with zero rain:\n{depth}"


def test_rain_produces_positive_depth(east_slope_session):
    """Rainfall on a slope must create positive water depth somewhere."""
    depth = run_sim(east_slope_session)
    assert np.sum(depth) > 0, "Expected positive total depth with rainfall on a slope"


def test_more_rain_gives_more_depth(east_slope_session):
    """Doubling rainfall must increase total water depth.

    Manning's kinematic wave gives h proportional to R^(3/5), so doubling R
    increases depth by a factor of 2^(3/5).
    """
    sum_low = float(np.sum(run_sim(east_slope_session, rain_value=50)))
    sum_high = float(np.sum(run_sim(east_slope_session, rain_value=100)))
    assert sum_high > sum_low, (
        f"Total depth should increase with rainfall rate: "
        f"sum(rain=50)={sum_low:.3e}, sum(rain=100)={sum_high:.3e}"
    )
    assert sum_high / sum_low == pytest.approx(2 ** (3 / 5), rel=0.3)


def test_higher_manning_gives_more_depth(east_slope_session):
    """Rougher surface (higher Manning's n) must increase water depth."""
    sum_smooth = float(np.sum(run_sim(east_slope_session, man_value=0.05)))
    sum_rough = float(np.sum(run_sim(east_slope_session, man_value=0.5)))
    assert sum_rough > sum_smooth, (
        f"Total depth should increase with Manning's n: "
        f"sum(n=0.05)={sum_smooth:.3e}, sum(n=0.5)={sum_rough:.3e}"
    )


def test_infiltration_reduces_depth(east_slope_session):
    """Adding overland-flow infiltration must reduce water depth.

    Infiltration removes water from the flowing sheet, decreasing both
    depth and discharge.
    """
    sum_no_infil = float(
        np.sum(run_sim(east_slope_session, rain_value=RAIN, infil_value=0))
    )
    sum_with_infil = float(
        np.sum(run_sim(east_slope_session, rain_value=RAIN, infil_value=RAIN // 2))
    )
    assert sum_no_infil > sum_with_infil, (
        f"Total depth should decrease when infiltration is added: "
        f"sum(no infil)={sum_no_infil:.3e}, sum(with infil)={sum_with_infil:.3e}"
    )


def test_depth_increases_downstream(tmp_path):
    """Water must be deeper in the downslope half of a uniform slope.

    By continuity, discharge q(x) = R * x grows linearly with distance x
    from the divide. Manning's kinematic wave then gives h(x) proportional
    to x^(3/5), also increasing downstream.

    Uses a 1-row x 6-column domain (instead of the standard 5-column one)
    so the strip splits cleanly into an upslope half (cols 0-2) and a
    downslope half (cols 3-5).
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

        flat = run_sim(session).flatten()  # shape (6,) for a 1-row raster
        upslope_sum = float(np.sum(flat[:3]))
        downslope_sum = float(np.sum(flat[3:]))
        assert downslope_sum > upslope_sum, (
            f"Downslope total depth ({downslope_sum:.3e}) should exceed "
            f"upslope total depth ({upslope_sum:.3e})"
        )


def test_steeper_slope_gives_less_depth(tmp_path):
    """A steeper slope must produce shallower water depth for the same rainfall.

    Manning's kinematic wave gives h proportional to S^(-3/10), so doubling
    the slope magnitude S reduces depth by a factor of 2^(-3/10).

    Elevation and dx are kept consistent: a 1 m/cell drop uses dx=1, and a
    2 m/cell drop uses dx=2, matching the slope magnitude S in Manning's
    equation.
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
            return np.asarray(run_sim(session))

    # gentle: drops 1 m per cell (elevation 5,4,3,2,1), dx = 1
    sum_gentle = float(np.sum(setup_and_run(tmp_path / "gentle", "6 - col()", 1)))
    # steep: drops 2 m per cell (elevation 9,7,5,3,1), dx = 2
    sum_steep = float(np.sum(setup_and_run(tmp_path / "steep", "11 - 2 * col()", 2)))

    assert sum_gentle > sum_steep, (
        f"Gentler slope depth ({sum_gentle:.3e}) should exceed "
        f"steeper slope depth ({sum_steep:.3e})"
    )
    assert sum_gentle / sum_steep == pytest.approx(2 ** (3 / 10), rel=0.3)


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


def test_results_consistent_across_seeds(east_slope_session):
    """Different random seeds must produce similar total depth.

    The Monte Carlo result converges as nwalkers grows. With 10000 walkers
    the total depth should be stable across seeds within a few percent.
    """
    seeds = [1, 7, 42, 99, 123]
    sums = [float(np.sum(run_sim(east_slope_session, random_seed=s))) for s in seeds]
    mean_sum = np.mean(sums)
    for seed, total in zip(seeds, sums, strict=True):
        assert total == pytest.approx(mean_sum, rel=0.1), (
            f"Seed {seed} total depth {total:.3e} deviates from mean {mean_sum:.3e}"
        )


def test_rain_raster_matches_scalar(east_slope_session):
    """A uniform rain raster must produce the same depth as the equivalent scalar."""
    tools = Tools(session=east_slope_session)
    tools.r_mapcalc(expression=f"rain_map = {RAIN}")
    sum_scalar = float(np.sum(run_sim(east_slope_session, rain_value=RAIN)))
    sum_raster = float(
        np.sum(run_sim(east_slope_session, rain="rain_map", rain_value=None))
    )
    assert sum_raster == pytest.approx(sum_scalar, rel=1e-6)


def test_random_seed_flag(east_slope_session):
    """The -s flag must generate a random seed and produce valid, varying output.

    Two runs with -s should both produce positive depth but differ from each
    other because each run gets a different auto-generated seed.
    """
    tools = Tools(session=east_slope_session)
    common = {
        "elevation": "elevation",
        "dx": "dx",
        "dy": "dy",
        "depth": np.array,
        "rain_value": RAIN,
        "man_value": 0.1,
        "nwalkers": NWALKERS,
        "niterations": NITERATIONS,
        "nprocs": NPROCS,
        "flags": "s",
    }
    depth_a = np.asarray(tools.r_sim_water(**common))
    depth_b = np.asarray(tools.r_sim_water(**common))
    assert np.sum(depth_a) > 0, "Expected positive depth with -s flag (run a)"
    assert np.sum(depth_b) > 0, "Expected positive depth with -s flag (run b)"
    assert not np.array_equal(depth_a, depth_b), (
        "Two runs with -s should produce different results"
    )


def run_sim_error(session, *, nwalkers=NWALKERS, **kwargs):
    """Run r.sim.water and return the error output as ndarray."""
    defaults = {"rain_value": RAIN, "infil_value": 0, "man_value": 0.1}
    defaults.update(kwargs)
    defaults = {k: v for k, v in defaults.items() if v is not None}
    tools = Tools(session=session)
    return np.asarray(
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            error=np.array,
            nwalkers=nwalkers,
            niterations=NITERATIONS,
            random_seed=SEED,
            nprocs=NPROCS,
            **defaults,
        )
    )


def test_error_output(east_slope_session):
    """The error output must be non-negative."""
    error = run_sim_error(east_slope_session)
    assert np.all(error >= 0), f"Error output should be non-negative:\n{error}"


def test_no_rain_produces_no_error(east_slope_session):
    """Zero rainfall must yield zero error everywhere."""
    error = run_sim_error(east_slope_session, rain_value=0)
    assert np.all(error == 0), f"Expected all-zero error with zero rain:\n{error}"


def test_more_walkers_reduces_error(east_slope_session):
    """More walkers must reduce the Monte Carlo sampling error.

    The error scales as 1/sqrt(N), so quadrupling walkers should
    roughly halve the total error.
    """
    error_few = float(np.sum(run_sim_error(east_slope_session, nwalkers=NWALKERS)))
    error_many = float(np.sum(run_sim_error(east_slope_session, nwalkers=4 * NWALKERS)))
    assert error_many < error_few, (
        f"Total error should decrease with more walkers: "
        f"sum({NWALKERS})={error_few:.3e}, sum({4 * NWALKERS})={error_many:.3e}"
    )


def test_mintimestep(east_slope_session):
    """A larger minimum time step must still produce valid positive depth.

    Increasing mintimestep reduces accuracy but speeds up the simulation.
    The result should still be positive and in the same order of magnitude.
    """
    sum_default = float(np.sum(run_sim(east_slope_session)))
    sum_large_step = float(np.sum(run_sim(east_slope_session, mintimestep=1.0)))
    assert sum_large_step > 0, "Expected positive depth with large mintimestep"
    # With 1 m cells, a 1-second minimum step lets walkers overshoot multiple
    # cells, roughly doubling the total depth. On 10 m cells the same step has
    # negligible effect (~2%). We only check same order of magnitude here.
    assert sum_large_step == pytest.approx(sum_default, rel=2.0), (
        f"Large mintimestep result ({sum_large_step:.3e}) deviates too far "
        f"from default ({sum_default:.3e})"
    )


def test_higher_diffusion_coeff_reduces_depth(east_slope_session):
    """Higher diffusion coefficient must reduce total water depth.

    The diffusion term spreads walkers away from flow concentration zones.
    More diffusion means walkers disperse faster, reducing depth accumulation.
    """
    sum_low = float(np.sum(run_sim(east_slope_session, diffusion_coeff=0.2)))
    sum_high = float(np.sum(run_sim(east_slope_session, diffusion_coeff=2.0)))
    assert sum_low > sum_high, (
        f"Total depth should decrease with higher diffusion: "
        f"sum(dc=0.2)={sum_low:.3e}, sum(dc=2.0)={sum_high:.3e}"
    )


def test_lower_hmax_increases_diffusion(tmp_path):
    """A lower hmax threshold must increase diffusion, reducing total depth.

    When water depth exceeds hmax, diffusion is amplified by (halpha + 1).
    Lowering hmax causes this amplification to kick in sooner.

    Uses a 200-cell domain with high rain and roughness so that depths
    exceed the hmax threshold.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=200, s=0, n=1, res=1)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        nw = 400000
        sum_low_hmax = float(
            np.sum(
                run_sim(
                    session, rain_value=1000, man_value=0.5, hmax=0.001, nwalkers=nw
                )
            )
        )
        sum_default_hmax = float(
            np.sum(
                run_sim(session, rain_value=1000, man_value=0.5, hmax=0.3, nwalkers=nw)
            )
        )
        assert sum_low_hmax < sum_default_hmax, (
            f"Lower hmax should increase diffusion and reduce depth: "
            f"sum(hmax=0.001)={sum_low_hmax:.3e}, sum(hmax=0.3)={sum_default_hmax:.3e}"
        )


def test_higher_halpha_reduces_depth(tmp_path):
    """A higher halpha must increase the diffusion boost above hmax.

    halpha controls how much extra diffusion is applied when depth exceeds
    hmax: diffusion is multiplied by (halpha + 1). A low hmax is needed
    so that depths actually exceed the threshold.

    Uses a 200-cell domain with high rain and roughness.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=200, s=0, n=1, res=1)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        nw = 400000
        sum_low = float(
            np.sum(
                run_sim(
                    session,
                    rain_value=1000,
                    man_value=0.5,
                    hmax=0.001,
                    halpha=0.5,
                    nwalkers=nw,
                )
            )
        )
        sum_high = float(
            np.sum(
                run_sim(
                    session,
                    rain_value=1000,
                    man_value=0.5,
                    hmax=0.001,
                    halpha=50.0,
                    nwalkers=nw,
                )
            )
        )
        assert sum_high < sum_low, (
            f"Higher halpha should increase diffusion and reduce depth: "
            f"sum(halpha=0.5)={sum_low:.3e}, sum(halpha=50)={sum_high:.3e}"
        )


def test_hbeta_changes_result(tmp_path):
    """Changing hbeta must produce a different result when depth exceeds hmax.

    hbeta weights the running average of walker velocity above the hmax
    threshold. The effect is small on a 1D domain, so only inequality
    (not direction) is checked.

    Uses a 200-cell domain with high rain, roughness, and low hmax.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=200, s=0, n=1, res=1)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        nw = 400000
        depth_default = run_sim(
            session, rain_value=1000, man_value=0.5, hmax=0.001, hbeta=0.5, nwalkers=nw
        )
        depth_high = run_sim(
            session, rain_value=1000, man_value=0.5, hmax=0.001, hbeta=10.0, nwalkers=nw
        )
        assert not np.array_equal(depth_default, depth_high), (
            "Changing hbeta should produce a different depth result"
        )


def test_man_raster_matches_scalar(east_slope_session):
    """A uniform Manning's n raster must produce the same depth as the equivalent scalar."""
    man_n = 0.1
    tools = Tools(session=east_slope_session)
    tools.r_mapcalc(expression=f"man_map = {man_n}")
    sum_scalar = float(np.sum(run_sim(east_slope_session, man_value=man_n)))
    sum_raster = float(
        np.sum(run_sim(east_slope_session, man="man_map", man_value=None))
    )
    assert sum_raster == pytest.approx(sum_scalar, rel=1e-6)


def test_infil_raster_matches_scalar(east_slope_session):
    """A uniform infiltration raster must produce the same depth as the equivalent scalar."""
    tools = Tools(session=east_slope_session)
    infil_rate = RAIN // 2
    tools.r_mapcalc(expression=f"infil_map = {infil_rate}")
    sum_scalar = float(np.sum(run_sim(east_slope_session, infil_value=infil_rate)))
    sum_raster = float(
        np.sum(run_sim(east_slope_session, infil="infil_map", infil_value=None))
    )
    assert sum_raster == pytest.approx(sum_scalar, rel=1e-6)


def test_dx_dy_optional(tmp_path):
    """Omitting dx and dy must produce valid positive depth.

    The documentation states that dx and dy are optional; when omitted,
    partial derivatives are computed internally from the elevation.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=5, s=0, n=1, res=1)
        tools.r_mapcalc(expression="elevation = 6 - col()")
        depth = np.asarray(
            tools.r_sim_water(
                elevation="elevation",
                depth=np.array,
                rain_value=RAIN,
                man_value=0.1,
                nwalkers=NWALKERS,
                niterations=NITERATIONS,
                random_seed=SEED,
                nprocs=NPROCS,
            )
        )
        assert np.sum(depth) > 0, "Expected positive depth when dx/dy are omitted"


def test_flow_control_increases_depth(east_slope_session):
    """A uniform flow control map must increase total water depth.

    The flow_control raster defines a per-cell trapping probability (0-1).
    When trapped, a walker's velocity is reversed and scaled to 10%,
    slowing it down. Slower walkers linger longer, increasing depth.
    Higher trapping probability means more accumulation.
    """
    tools = Tools(session=east_slope_session)
    sum_no_control = float(np.sum(run_sim(east_slope_session)))
    tools.r_mapcalc(expression="flow_ctrl_low = 0.3")
    tools.r_mapcalc(expression="flow_ctrl_high = 0.8")
    sum_low_trap = float(
        np.sum(run_sim(east_slope_session, flow_control="flow_ctrl_low"))
    )
    sum_high_trap = float(
        np.sum(run_sim(east_slope_session, flow_control="flow_ctrl_high"))
    )
    assert sum_low_trap > sum_no_control, (
        f"Trapping should increase depth: "
        f"sum(no trap)={sum_no_control:.3e}, sum(trap=0.3)={sum_low_trap:.3e}"
    )
    assert sum_high_trap > sum_low_trap, (
        f"More trapping should increase depth further: "
        f"sum(trap=0.3)={sum_low_trap:.3e}, sum(trap=0.8)={sum_high_trap:.3e}"
    )


def test_time_series_output(tmp_path):
    """The -t flag with output_step must produce intermediate depth maps.

    With niterations=10 and output_step=5, the tool should create depth
    maps at minutes 5 and 10. The later time step should have greater or
    equal total depth as water accumulates over time.

    Uses a 200-cell domain at 10 m resolution with high roughness so that
    walkers survive the full simulation. On small or steep domains walkers
    leave before the first output step, producing no time-series maps.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=2000, s=0, n=10, res=10)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth="ts_depth",
            rain_value=RAIN,
            man_value=0.3,
            nwalkers=NWALKERS,
            niterations=NITERATIONS,
            output_step=5,
            random_seed=SEED,
            nprocs=NPROCS,
            flags="t",
        )

        sum_05 = tools.r_univar(map="ts_depth.05", format="json")["sum"]
        sum_10 = tools.r_univar(map="ts_depth.10", format="json")["sum"]
        assert sum_05 > 0, "Expected positive depth at t=5"
        assert sum_10 >= sum_05, (
            f"Depth at t=10 ({sum_10:.3e}) should be >= depth at t=5 ({sum_05:.3e})"
        )
