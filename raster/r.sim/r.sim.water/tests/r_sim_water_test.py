"""Minimal analytical tests for r.sim.water (SIMWE).

Tests use small, hand-crafted domains where expected output can be reasoned
about from first principles rather than from pre-computed reference values.
"""

import io
import os
import pathlib

import numpy as np
import pytest

import grass.script as gs
from grass.tools import Tools

# Fixed seed and single thread make results fully deterministic.
SEED = 42
NPROCS = 1

# With default walkers and niterations=2, small domains reach
# near-steady state in ~0.2 s per run while keeping Monte Carlo noise low
# enough for structural assertions.
NITERATIONS = 2  # minutes; sufficient for near-steady state on a small domain
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
        "nprocs": NPROCS,
        "niterations": NITERATIONS,
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
        random_seed=random_seed,
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
    """Rainfall on a slope must create positive water depth somewhere.

    See test_north_slope_rain_produces_positive_depth for the north-south variant.
    """
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

    Uses a 1-row x 6-column domain to split cleanly into an upslope half
    (cols 0-2) and a downslope half (cols 3-5). See
    test_north_slope_depth_increases_downstream for the north-south variant.
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
            niterations=NITERATIONS,
            random_seed=SEED,
            nprocs=NPROCS,
        )
    )
    assert np.sum(discharge) > 0, "Expected positive discharge with rainfall on a slope"


def test_results_consistent_across_seeds(east_slope_session):
    """Different random seeds must produce similar total depth.

    The Monte Carlo result converges as nwalkers grows. With enough walkers
    the total depth should be stable across seeds within a few percent.
    """
    seeds = [1, 7, 42, 99, 123]
    sums = [
        float(np.sum(run_sim(east_slope_session, nwalkers=1000, random_seed=s)))
        for s in seeds
    ]
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


def run_sim_error(session, **kwargs):
    """Run r.sim.water and return the error output as ndarray."""
    defaults = {
        "rain_value": RAIN,
        "infil_value": 0,
        "man_value": 0.1,
        "niterations": NITERATIONS,
    }
    defaults.update(kwargs)
    defaults = {k: v for k, v in defaults.items() if v is not None}
    tools = Tools(session=session)
    return np.asarray(
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            error=np.array,
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
    few = 10000
    many = 40000
    error_few = float(np.sum(run_sim_error(east_slope_session, nwalkers=few)))
    error_many = float(np.sum(run_sim_error(east_slope_session, nwalkers=many)))
    assert error_many < error_few, (
        f"Total error should decrease with more walkers: "
        f"sum({few})={error_few:.3e}, sum({many})={error_many:.3e}"
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


def test_longer_simulation_larger_domain(tmp_path):
    """More iterations must increase total water depth on a larger domain.

    On a larger domain with slower drainage, longer simulations accumulate
    more water. Uses a 200-cell domain at 10 m resolution so that walkers
    remain within the domain for the full duration.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=2000, s=0, n=10, res=10)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        sum_short = float(
            np.sum(run_sim(session, rain_value=RAIN, man_value=0.3, niterations=5))
        )
        sum_long = float(
            np.sum(run_sim(session, rain_value=RAIN, man_value=0.3, niterations=20))
        )
        assert sum_long > sum_short, (
            f"Longer simulation should produce more depth: "
            f"sum(5 min)={sum_short:.3e}, sum(20 min)={sum_long:.3e}"
        )
        ratio = sum_long / sum_short
        assert ratio >= 1.05, (
            f"20-min simulation should produce at least 5% more depth than 5-min "
            f"(ratio={ratio:.2f})"
        )


def test_niterations_affects_time_series_progression(tmp_path):
    """More iterations must create more time-series output maps.

    With output_step=5, niterations=10 produces maps at t=5,10 while
    niterations=20 produces maps at t=5,10,15. More time-series maps
    indicate longer simulation duration.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=2000, s=0, n=10, res=10)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        # niterations=10 with output_step=5 produces maps at t=5,10
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth="depth_10min",
            rain_value=RAIN,
            man_value=0.3,
            nwalkers=10000,
            niterations=10,
            output_step=5,
            random_seed=SEED,
            nprocs=NPROCS,
            flags="t",
        )

        # niterations=20 with output_step=5 produces maps at t=5,10,15
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth="depth_20min",
            rain_value=RAIN,
            man_value=0.3,
            nwalkers=10000,
            niterations=20,
            output_step=5,
            random_seed=SEED,
            nprocs=NPROCS,
            flags="t",
        )

        # 10-min run should produce 2 time-series maps (t=5, t=10)
        maps_10 = list(
            tools.g_list(type="raster", pattern="depth_10min*", format="json")
        )
        assert len(maps_10) == 2, (
            f"10-min simulation with output_step=5 should produce 2 maps, got {len(maps_10)}"
        )

        # 20-min run should produce 3 time-series maps (t=5, t=10, t=15)
        maps_20 = list(
            tools.g_list(type="raster", pattern="depth_20min*", format="json")
        )
        assert len(maps_20) == 3, (
            f"20-min simulation with output_step=5 should produce 3 maps, got {len(maps_20)}"
        )
        assert len(maps_20) > len(maps_10), (
            f"Longer simulation should produce more time-series maps: "
            f"20-min has {len(maps_20)} maps, 10-min has {len(maps_10)} maps"
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
            niterations=10,
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


def test_observation_logfile(east_slope_session, tmp_path):
    """Observation points must log water depth at each time step.

    Three observation points are placed on the east_slope_session domain
    at upslope, midslope, and downslope positions. The logfile must contain
    a header with category numbers and data lines with depth values.
    Depth should increase from upslope to downslope. See
    test_north_slope_observation_logfile for the north-south variant.
    """
    tools = Tools(session=east_slope_session)

    points_data = io.StringIO("0.5|0.5|1\n2.5|0.5|2\n4.5|0.5|3\n")
    tools.v_in_ascii(input=points_data, output="points", cat=3)

    logfile = str(tmp_path / "obs_log.txt")
    tools.r_sim_water(
        elevation="elevation",
        dx="dx",
        dy="dy",
        depth=np.array,
        rain_value=RAIN,
        man_value=0.1,
        nwalkers=1000,
        niterations=NITERATIONS,
        random_seed=SEED,
        nprocs=NPROCS,
        observation="points",
        logfile=logfile,
    )

    lines = pathlib.Path(logfile).read_text(encoding="utf-8").strip().split("\n")

    # Header: "STEP   CAT0001 CAT0002 CAT0003"
    header = lines[0].split()
    assert header[0] == "STEP"
    assert "CAT0001" in header
    assert "CAT0002" in header
    assert "CAT0003" in header

    # Must have at least one data line after the header.
    assert len(lines) > 1, "Logfile should contain data lines after the header"

    # Data line: "000028 0.0000 0.0001 0.0001" (step, then one depth per point)
    last_vals = [float(v) for v in lines[-1].split()[1:]]
    upslope = last_vals[0]
    midslope = last_vals[1]
    downslope = last_vals[2]
    assert downslope >= midslope >= upslope, (
        f"Depth should increase downstream: "
        f"upslope={upslope:.4f}, midslope={midslope:.4f}, downslope={downslope:.4f}"
    )


def test_walkers_output(tmp_path):
    """The walkers_output parameter must produce a vector point map.

    Each surviving walker position is written as a 3D point. The number
    of points should not exceed nwalkers.

    Uses a large domain (200 cells) to give walkers sufficient residence
    time before exiting. High roughness (0.3) further increases walker
    retention; small domains produce no walker output regardless of roughness.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=2000, s=0, n=10, res=10)
        tools.r_mapcalc(expression="elevation = 201 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")

        nwalkers = 500
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth=np.array,
            rain_value=RAIN,
            man_value=0.3,
            nwalkers=nwalkers,
            niterations=NITERATIONS,
            random_seed=SEED,
            nprocs=NPROCS,
            walkers_output="walkers",
        )

        info = tools.v_info(map="walkers", flags="t", format="json")
        npoints = int(info["points"])
        assert npoints > 0, "Expected at least one walker point"
        assert npoints <= nwalkers, (
            f"Number of walker points ({npoints}) should not exceed "
            f"nwalkers ({nwalkers})"
        )


def test_walkers_output_time_series(tmp_path):
    """With -t, walkers_output must produce per-step vector maps.

    The maps use an underscore-separated time suffix (e.g., walkers_05)
    unlike raster outputs which use a dot (e.g., depth.05).

    Uses a large domain (200 cells) with high roughness (0.3) so walkers
    survive long enough to generate multiple time-series outputs.
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
            niterations=10,
            output_step=5,
            random_seed=SEED,
            nprocs=NPROCS,
            walkers_output="walkers",
            flags="t",
        )

        info_05 = tools.v_info(map="walkers_05", flags="t", format="json")
        info_10 = tools.v_info(map="walkers_10", flags="t", format="json")
        assert int(info_05["points"]) > 0, "Expected walker points at t=5"
        assert int(info_10["points"]) > 0, "Expected walker points at t=10"


def test_nprocs_gives_same_result(east_slope_session):
    """Multiple threads produce a result close to a single thread.

    Thread-level partitioning changes walker ordering and random draws,
    so results are not bitwise identical, but total depth should agree
    within a few percent.
    """
    sum_single = float(np.sum(run_sim(east_slope_session, nwalkers=1000)))
    sum_multi = float(np.sum(run_sim(east_slope_session, nwalkers=1000, nprocs=4)))
    tolerance = 0.05
    assert sum_multi == pytest.approx(sum_single, rel=tolerance), (
        f"nprocs=4 result ({sum_multi:.3e}) should match "
        f"nprocs=1 result ({sum_single:.3e}) within {tolerance:.0%}"
    )


def test_north_slope_rain_produces_positive_depth(tmp_path):
    """Rainfall on a north-south slope must create positive water depth.

    Mirrors test_rain_produces_positive_depth with dy instead of dx, verifying
    flow direction when dy != 0 and dx = 0.
    """
    project = tmp_path / "simwe_north"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=1, s=0, n=5, res=1)
        tools.r_mapcalc(expression="elevation = row()")
        tools.r_mapcalc(expression="dx = 0.0")
        tools.r_mapcalc(expression="dy = 1.0")

        depth = run_sim(session)
        assert np.sum(depth) > 0, (
            "Expected positive total depth with rainfall on north-south slope"
        )


def test_north_slope_depth_increases_downstream(tmp_path):
    """Water must be deeper in the downslope (northward) half of a north-south slope.

    By continuity, discharge q(y) = R * y grows linearly with distance y
    from the divide. Manning's kinematic wave then gives h(y) proportional
    to y^(3/5), increasing toward the northern boundary.

    Uses a 6-row x 1-column domain to split cleanly into a downslope half
    (rows 0-2, north) and an upslope half (rows 3-5, south). Mirrors
    test_depth_increases_downstream with dy instead of dx.

    Note: dy = 1.0 because elevation increases southward (with increasing row).
    """
    project = tmp_path / "simwe_north_grad"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=1, s=0, n=6, res=1)
        # elevation = row() gives [1, 2, 3, 4, 5, 6] north to south
        # North (low row index) has low elevation, water flows north
        tools.r_mapcalc(expression="elevation = row()")
        tools.r_mapcalc(expression="dx = 0.0")
        tools.r_mapcalc(expression="dy = 1.0")

        depth = run_sim(session).flatten()  # shape (6,) for a 1-column raster
        upslope_sum = float(np.sum(depth[3:]))  # rows 3-5 (south, high elevation)
        downslope_sum = float(np.sum(depth[:3]))  # rows 0-2 (north, low elevation)
        assert downslope_sum > upslope_sum, (
            f"Downslope total depth ({downslope_sum:.3e}) should exceed "
            f"upslope total depth ({upslope_sum:.3e})"
        )


def test_north_slope_observation_logfile(tmp_path):
    """Observation parameter works with north-south (dy != 0) flow.

    Mirrors test_observation_logfile with dy instead of dx. Water flows north
    (downhill), so observation points should show increasing depth from south
    (cat0001) to north (cat0003). Uses a 10-row x 1-column domain for adequate
    water flow to observation points.
    """
    project = tmp_path / "simwe_north_obs"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=1, s=0, n=10, res=1)
        tools.r_mapcalc(expression="elevation = row()")
        tools.r_mapcalc(expression="dx = 0.0")
        tools.r_mapcalc(expression="dy = 1.0")

        # Three observation points along the north-south axis.
        points_data = io.StringIO("0.5|1.5|1\n0.5|5.5|2\n0.5|9.5|3\n")
        tools.v_in_ascii(input=points_data, output="points", cat=3)

        logfile = str(tmp_path / "obs_log_north.txt")
        tools.r_sim_water(
            elevation="elevation",
            dx="dx",
            dy="dy",
            depth=np.array,
            rain_value=RAIN,
            man_value=0.1,
            nwalkers=1000,
            niterations=NITERATIONS,
            random_seed=SEED,
            nprocs=NPROCS,
            observation="points",
            logfile=logfile,
        )

        lines = pathlib.Path(logfile).read_text(encoding="utf-8").strip().split("\n")

        # Header: "STEP   CAT0001 CAT0002 CAT0003"
        header = lines[0].split()
        assert header[0] == "STEP"
        assert "CAT0001" in header
        assert "CAT0002" in header
        assert "CAT0003" in header

        # Must have at least one data line after the header.
        assert len(lines) > 1, "Logfile should contain data lines after the header"

        # Data line: depth increases from south (cat0001) to north (cat0003)
        last_vals = [float(v) for v in lines[-1].split()[1:]]
        south = last_vals[0]
        mid = last_vals[1]
        north = last_vals[2]
        assert north >= mid >= south, (
            f"Depth should increase downstream (northward): "
            f"south={south:.4f}, mid={mid:.4f}, north={north:.4f}"
        )
