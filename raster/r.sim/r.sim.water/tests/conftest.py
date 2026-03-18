import os

import pytest

import grass.script as gs
from grass.experimental.mapset import TemporaryMapsetSession
from grass.tools import Tools


@pytest.fixture(scope="module")
def simwe_session(tmp_path_factory):
    """Module-scoped GRASS session for SIMWE tests with shared base terrain.

    Creates a persistent project and initializes the GRASS runtime environment
    in the PERMANENT mapset, where the base terrain (elevation, dx, dy) is set up.
    This session persists for the entire test module. All function-scoped fixtures
    create temporary mapsets within this session and automatically see the base
    terrain from PERMANENT.
    """
    tmp_path = tmp_path_factory.mktemp("simwe_module")
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        # Set up base 1x5 eastward slope terrain in PERMANENT mapset.
        # Use -s flag to save region as default (inherited by temporary mapsets).
        tools.g_region(w=0, e=5, s=0, n=1, res=1, flags="s")
        tools.r_mapcalc(expression="elevation = 6 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")
        yield session


@pytest.fixture
def east_slope_session(simwe_session):
    """Session with a 1-row x 5-column eastward slope, ready for r.sim.water.

    Uses the module-scoped simwe_session and creates a new temporary mapset
    for each test. The base terrain (elevation, dx, dy) and computational region
    are inherited from the PERMANENT mapset created in the module scope.

    Terrain (GRASS col() is 1-indexed from west):
      elevation = 6 - col()  -> [5, 4, 3, 2, 1] west to east
      dx = 1.0               (unit slope vector pointing east)
      dy = 0.0               (no north-south component)

    Water flows east and accumulates toward the eastern boundary.
    """
    with TemporaryMapsetSession(env=simwe_session.env) as session:
        yield session
