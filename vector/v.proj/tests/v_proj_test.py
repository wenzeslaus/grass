# SPDX-License-Identifier: GPL-2.0-or-later

"""Test v.proj coordinate correctness.

With smax=0 (no densification) v.proj transforms all vertices of a line
in one GPJ_transform_array() call, while with densification enabled each
vertex goes through a separate GPJ_transform() call. Points always go
through GPJ_transform_array(). Both paths must produce the same
coordinates.
"""

import pytest

from grass.tools import Tools

# Raleigh, NC area in NAD83(2011) as "x y" (longitude latitude).
NAD83_POINT_ASCII = """\
P 1 1
 -78.00 35.00
 1 1
"""

# Three vertices about 1 km apart, starting at the point above, so that
# the default smax (10000 m) adds no vertices.
NAD83_LINE_ASCII = """\
L 3 1
 -78.00 35.00
 -78.01 35.01
 -78.02 35.00
 1 1
"""

# The point above in ITRF2014 at the reference epoch of the transformation
# (cs2cs and GDAL without a coordinate epoch).
ITRF_X = -78.000004542
ITRF_Y = 35.000007465

# 1e-7 degrees is about 1 cm here.
DEGREE_TOLERANCE = 1e-7


@pytest.fixture
def nad83_to_itrf_session(make_reprojection_session):
    """Active session in an ITRF2014 project with a NAD83(2011) project next
    to it containing the vector maps "point" and "line".

    NAD83(2011) to ITRF2014 (EPSG:7807) is a Helmert transformation with
    time-dependent parameters.
    """
    return make_reprojection_session(
        "6318", "9000", {"point": NAD83_POINT_ASCII, "line": NAD83_LINE_ASCII}
    )


def read_coords(tools, vector_map, geometry_type="line"):
    """Return list of (x, y) vertex tuples of the features in the map"""
    ascii_out = tools.v_out_ascii(
        input=vector_map, format="standard", type=geometry_type
    )
    coords = []
    for line in ascii_out.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and "." in parts[0]:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def test_smax_zero_matches_per_vertex_path(reprojection_session):
    """All vertices must be reprojected, not just the first one"""
    tools = Tools(session=reprojection_session.session)
    tools.v_proj(
        project=reprojection_session.source, input="line", output="batch", smax=0
    )
    tools.v_proj(project=reprojection_session.source, input="line", output="per_vertex")
    batch = read_coords(tools, "batch")
    per_vertex = read_coords(tools, "per_vertex")
    assert len(batch) == 3
    assert batch == pytest.approx(per_vertex)


def test_point_uses_reference_epoch(nad83_to_itrf_session):
    """A point is reprojected at the reference epoch (GPJ_transform_array())"""
    tools = Tools(session=nad83_to_itrf_session.session)
    tools.v_proj(project=nad83_to_itrf_session.source, input="point", output="point")
    coords = read_coords(tools, "point", geometry_type="point")
    assert coords == [
        pytest.approx((ITRF_X, ITRF_Y), abs=DEGREE_TOLERANCE),
    ]


def test_line_uses_reference_epoch(nad83_to_itrf_session):
    """Line vertices are reprojected at the reference epoch (GPJ_transform())

    Only the first vertex has a reference value.
    """
    tools = Tools(session=nad83_to_itrf_session.session)
    tools.v_proj(project=nad83_to_itrf_session.source, input="line", output="line")
    coords = read_coords(tools, "line")
    assert len(coords) == 3
    assert coords[0] == pytest.approx((ITRF_X, ITRF_Y), abs=DEGREE_TOLERANCE)
