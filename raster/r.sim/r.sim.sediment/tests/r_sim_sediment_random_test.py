"""Tests of the random numbers of r.sim.sediment

Each walker draws from a random number stream of its own, determined by the
seed and the walker's number, so the draws do not depend on the number of
threads. The sediment flux still does at nprocs > 1, because walkers in the
same cell update the flux grid without synchronization, so the test which
compares different numbers of threads compares the walker positions instead.
A walker's path depends only on the slope and its random numbers.
"""

import os
from io import StringIO

import numpy as np
import pytest

import grass.script as gs
from grass.tools import Tools


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session in an XY project with a bowl as elevation and constant inputs"""
    project = tmp_path_factory.mktemp("r_sim_sediment_random") / "xy_test"
    gs.create_project(project)
    with (
        gs.setup.init(project, env=os.environ.copy()) as session,
        Tools(session=session) as tools,
    ):
        tools.g_region(s=0, n=40, w=0, e=40, res=1)
        tools.r_mapcalc(
            expression="elevation = 0.01 * ((col() - 20.5)^2 + (row() - 20.5)^2)"
        )
        tools.r_mapcalc(expression="water_depth = 0.1")
        tools.r_mapcalc(expression="detachment = 0.001")
        tools.r_mapcalc(expression="transport = 0.001")
        tools.r_mapcalc(expression="shear_stress = 0")
        yield session


def simulate(session, flux, walkers, **kwargs):
    """Run the simulation, return the flux and the walker coordinates"""
    tools = Tools(session=session)
    tools.r_sim_sediment(
        elevation="elevation",
        water_depth="water_depth",
        detachment_coeff="detachment",
        transport_coeff="transport",
        shear_stress="shear_stress",
        sediment_flux=flux,
        walkers_output=walkers,
        duration=2,
        **kwargs,
    )
    text = tools.v_out_ascii(input=walkers, format="point", precision=15).text
    positions = np.loadtxt(StringIO(text), delimiter="|")
    return gs.array.array(flux, env=session.env), positions


def test_same_seed_gives_same_flux(session):
    """Two runs with the same seed give the same sediment flux"""
    first, _ = simulate(session, "flux_seed_1_a", "walkers_seed_1_a", random_seed=1)
    second, _ = simulate(session, "flux_seed_1_b", "walkers_seed_1_b", random_seed=1)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_results(session):
    """Runs with different seeds give different fluxes and walker positions"""
    first_flux, first_walkers = simulate(
        session, "flux_seed_1", "walkers_seed_1", random_seed=1
    )
    second_flux, second_walkers = simulate(
        session, "flux_seed_2", "walkers_seed_2", random_seed=2
    )
    assert not np.array_equal(first_flux, second_flux)
    assert not np.array_equal(first_walkers, second_walkers)


@pytest.mark.parametrize("nprocs", [2, 4])
def test_walkers_do_not_depend_on_nprocs(session, nprocs):
    """Walkers end at the same positions with any number of threads"""
    _, serial = simulate(
        session, f"flux_{nprocs}_serial", f"walkers_{nprocs}_serial", random_seed=5
    )
    _, parallel = simulate(
        session,
        f"flux_{nprocs}_parallel",
        f"walkers_{nprocs}_parallel",
        random_seed=5,
        nprocs=nprocs,
    )
    assert serial.size
    assert np.array_equal(parallel, serial)
