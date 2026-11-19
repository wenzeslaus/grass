import os

import pytest

import grass.script as gs
from grass.experimental import TemporaryMapsetSession
from grass.tools import Tools


@pytest.fixture(scope="module")
def session_for_module(tmp_path_factory):
    """Module-scoped session with a raster and a possible mask"""
    tmp_path = tmp_path_factory.mktemp("data") / "project"
    gs.create_project(tmp_path)
    with gs.setup.init(tmp_path, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(n=20, s=0, e=20, w=0, rows=20, cols=20)
        tools.r_mapcalc(expression="data = row() + col()")
        tools.r_mapcalc(expression="raster_mask = if(row() < 10, 1, null())")
        yield session


@pytest.fixture
def session(session_for_module):
    """A session in a temporary mapset"""
    with TemporaryMapsetSession(env=session_for_module.env) as session:
        yield session


@pytest.fixture(scope="module")
def crs_session_for_module(tmp_path_factory):
    """Module-scoped session in a project with EPSG:3358

    Linking a file with r.external needs a project with a CRS.
    The project has only the default region of one cell.
    """
    project = tmp_path_factory.mktemp("crs") / "project"
    gs.create_project(project, epsg=3358)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        yield session


@pytest.fixture
def crs_session(crs_session_for_module):
    """A session in a temporary mapset of the project with a CRS

    The mapset starts with the default region of one cell, so a test
    needs to set the region it works in.
    """
    with TemporaryMapsetSession(env=crs_session_for_module.env) as session:
        yield session
