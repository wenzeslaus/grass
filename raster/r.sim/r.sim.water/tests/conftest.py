import os

import pytest

import grass.script as gs
from grass.tools import Tools


@pytest.fixture
def east_slope_session(tmp_path):
    """Session with a 1-row x 5-column eastward slope, ready for r.sim.water.

    Terrain (GRASS col() is 1-indexed from west):
      elevation = 6 - col()  -> [5, 4, 3, 2, 1] west to east
      dx = 1.0               (unit slope vector pointing east)
      dy = 0.0               (no north-south component)

    Water flows east and accumulates toward the eastern boundary.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=5, s=0, n=1, res=1)
        tools.r_mapcalc(expression="elevation = 6 - col()")
        tools.r_mapcalc(expression="dx = 1.0")
        tools.r_mapcalc(expression="dy = 0.0")
        yield session
