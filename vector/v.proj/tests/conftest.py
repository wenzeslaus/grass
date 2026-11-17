# SPDX-License-Identifier: GPL-2.0-or-later

import os
from contextlib import ExitStack
from io import StringIO
from types import SimpleNamespace

import pytest

import grass.script as gs
from grass.tools import Tools

# One line with three vertices about 1 km apart, so that the default
# smax (10000 m) adds no vertices during reprojection.
LINE_ASCII = """\
L 3 1
 10.00 50.00
 10.01 50.01
 10.02 50.00
 1 1
"""


@pytest.fixture
def make_reprojection_session(tmp_path):
    """Return a function which creates a source and a target project

    The function takes the EPSG codes of the source and target projects
    and a dictionary of vector map names and their content in the standard
    v.in.ascii format, which are created in the source project. It returns
    an active session in the target project and the name of the source
    project.
    """
    with ExitStack() as stack:

        def make(source_epsg, target_epsg, vector_maps):
            source = "source"
            gs.create_project(tmp_path, source, epsg=source_epsg)
            with gs.setup.init(tmp_path / source, env=os.environ.copy()) as session:
                tools = Tools(session=session)
                for name, text in vector_maps.items():
                    tools.v_in_ascii(
                        input=StringIO(text), output=name, format="standard", flags="n"
                    )
            target = tmp_path / "target"
            gs.create_project(target, epsg=target_epsg)
            session = stack.enter_context(gs.setup.init(target, env=os.environ.copy()))
            return SimpleNamespace(session=session, source=source)

        yield make


@pytest.fixture
def reprojection_session(make_reprojection_session):
    """Active session in an EPSG:3857 project, with a lat/lon source
    project next to it containing a three-vertex line vector named "line".
    """
    return make_reprojection_session("4326", "3857", {"line": LINE_ASCII})
