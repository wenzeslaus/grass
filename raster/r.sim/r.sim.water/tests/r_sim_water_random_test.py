"""Tests of the random numbers of r.sim.water

Each walker draws from a random number stream of its own, determined by the
seed and the walker's number, so the draws do not depend on the number of
threads. The water depth still does at nprocs > 1, because walkers in the same
cell update the depth grid without synchronization, so the tests which compare
different numbers of threads compare the walker positions instead. With a
large hmax and no infiltration, a walker's path depends only on the slope and
its random numbers, not on the water depth. The elevation is a bowl, so that
the walkers stay in the region until the end of the simulation.
"""

import os
from io import StringIO

import numpy as np
import pytest

import grass.script as gs
from grass.exceptions import CalledModuleError
from grass.tools import Tools


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session in an XY project with a bowl as elevation"""
    project = tmp_path_factory.mktemp("r_sim_water_random") / "xy_test"
    gs.create_project(project)
    with (
        gs.setup.init(project, env=os.environ.copy()) as session,
        Tools(session=session) as tools,
    ):
        tools.g_region(s=0, n=40, w=0, e=40, res=1)
        tools.r_mapcalc(
            expression="elevation = 0.01 * ((col() - 20.5)^2 + (row() - 20.5)^2)"
        )
        yield session


def simulate_depth(session, name, **kwargs):
    """Run the simulation and return the water depth as an array"""
    tools = Tools(session=session)
    tools.r_sim_water(elevation="elevation", depth=name, duration=5, **kwargs)
    return gs.array.array(name, env=session.env)


def simulate_walkers(session, name, **kwargs):
    """Run the simulation and return the final walker coordinates as an array"""
    tools = Tools(session=session)
    tools.r_sim_water(elevation="elevation", walkers_output=name, duration=5, **kwargs)
    text = tools.v_out_ascii(input=name, format="point", precision=15).text
    return np.loadtxt(StringIO(text), delimiter="|")


def test_same_seed_gives_same_depth(session):
    """Two runs with the same seed give the same water depth"""
    first = simulate_depth(session, "depth_seed_1_a", random_seed=1)
    second = simulate_depth(session, "depth_seed_1_b", random_seed=1)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_depth(session):
    """Runs with different seeds give different water depths"""
    first = simulate_depth(session, "depth_seed_1", random_seed=1)
    second = simulate_depth(session, "depth_seed_2", random_seed=2)
    assert not np.array_equal(first, second)


def test_generated_seed_is_used_for_streams(session):
    """The -s flag seeds the streams with the seed it generates.

    GRASS_RANDOM_SEED sets the seed which the flag generates.
    """
    env = session.env.copy()
    env["GRASS_RANDOM_SEED"] = "3"
    tools = Tools(env=env)
    tools.r_sim_water(
        elevation="elevation", depth="depth_generated", duration=5, flags="s"
    )
    generated = gs.array.array("depth_generated", env=session.env)
    given = simulate_depth(session, "depth_given", random_seed=3)
    assert np.array_equal(generated, given)


@pytest.mark.parametrize("nprocs", [2, 4])
def test_walkers_do_not_depend_on_nprocs(session, nprocs):
    """Walkers end at the same positions with any number of threads"""
    serial = simulate_walkers(
        session, f"walkers_{nprocs}_serial", random_seed=5, hmax=1e6
    )
    parallel = simulate_walkers(
        session, f"walkers_{nprocs}_parallel", random_seed=5, hmax=1e6, nprocs=nprocs
    )
    assert serial.size
    assert np.array_equal(parallel, serial)


def test_walkers_depend_on_seed(session):
    """The walker comparison above can detect a difference"""
    first = simulate_walkers(session, "walkers_seed_5", random_seed=5, hmax=1e6)
    second = simulate_walkers(session, "walkers_seed_6", random_seed=6, hmax=1e6)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("seed", [-(2**31) - 1, 2**32])
def test_seed_outside_range_is_an_error(session, seed):
    """A seed the generator cannot use is refused, not silently wrapped"""
    tools = Tools(session=session)
    with pytest.raises(CalledModuleError, match="outside the range"):
        tools.r_sim_water(
            elevation="elevation",
            depth="depth_bad_seed",
            duration=1,
            random_seed=seed,
            overwrite=True,
        )
