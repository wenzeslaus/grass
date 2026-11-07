"""Tests of linked and virtual raster maps

Maps linked with r.external, maps written through r.external.out, and
virtual rasters from r.buildvrt are read through the raster library like
native maps. The tests write and overwrite such maps and check the values
read back, computed with NumPy from the expressions which created them.
"""

import numpy as np
import pytest

from grass.script import array as garray
from grass.tools import Tools

ROWS = 30
COLS = 20


def values(factor):
    """Values of the map factor * (row * 100 + col), rows and cols from 1"""
    row, col = np.mgrid[1 : ROWS + 1, 1 : COLS + 1]
    return factor * (row * 100.0 + col)


@pytest.fixture
def session(crs_session):
    """A session in a project with a CRS, with the region of the maps"""
    Tools(session=crs_session).g_region(n=ROWS, s=0, w=0, e=COLS, res=1)
    return crs_session


def read(session, name):
    """Values of a map as float64 with NaN for nulls"""
    return np.array(garray.array(name, null=np.nan, dtype=np.float64, env=session.env))


def write_geotiff(tools, tmp_path, factor):
    """Write values(factor) to a new GeoTIFF file and return its path"""
    tools.r_mapcalc(expression=f"source_{factor} = {factor} * (row() * 100 + col())")
    path = tmp_path / f"source_{factor}.tif"
    tools.r_out_gdal(input=f"source_{factor}", output=path, type="Float64")
    return path


def file_values(tools, session, path):
    """Values of the GeoTIFF file at path, read through a new link"""
    tools.r_external(input=path, output="file_check")
    return read(session, "file_check")


def test_relink(session, tmp_path):
    """A map linked with r.external and linked again with --overwrite has the values of the file"""
    tools = Tools(session=session)
    tools.r_external(input=write_geotiff(tools, tmp_path, 1), output="linked")
    assert np.array_equal(read(session, "linked"), values(1))
    tools.r_external(
        input=write_geotiff(tools, tmp_path, 2), output="linked", overwrite=True
    )
    assert np.array_equal(read(session, "linked"), values(2))


@pytest.mark.parametrize(
    ("output_format", "function"), [("GTiff", "float"), ("COG", "double")]
)
def test_write_through_external_out(session, tmp_path, output_format, function):
    """A floating point map with nulls is written through r.external.out

    GTiff is created directly, while COG is written to a GDAL memory
    dataset first and copied to the file when the map is closed.
    """
    tools = Tools(session=session)
    directory = tmp_path / "out"
    directory.mkdir()
    tools.r_external_out(directory=directory, format=output_format, extension="tif")
    value = f"{function}(3 * (row() * 100 + col()))"
    tools.r_mapcalc(expression=f"written = if(row() == col(), null(), {value})")
    assert (directory / "written.tif").exists()
    row, col = np.mgrid[1 : ROWS + 1, 1 : COLS + 1]
    expected = np.where(row == col, np.nan, values(3))
    assert np.array_equal(read(session, "written"), expected, equal_nan=True)


def test_overwrite_through_external_out(session, tmp_path):
    """A map written again through r.external.out has the new values in the file"""
    tools = Tools(session=session)
    tools.r_external_out(directory=tmp_path, format="GTiff", extension="tif")
    tools.r_mapcalc(expression="written = row() * 100 + col()")
    tools.r_mapcalc(expression="written = 4 * (row() * 100 + col())", overwrite=True)
    assert np.array_equal(read(session, "written"), values(4))
    assert np.array_equal(
        file_values(tools, session, tmp_path / "written.tif"), values(4)
    )


def test_native_map_through_external_out(session, tmp_path):
    """A native map overwritten through r.external.out is written to the file"""
    tools = Tools(session=session)
    tools.r_mapcalc(expression="map = row() * 100 + col()")
    tools.r_external_out(directory=tmp_path, format="GTiff", extension="tif")
    tools.r_mapcalc(expression="map = 5 * (row() * 100 + col())", overwrite=True)
    assert np.array_equal(read(session, "map"), values(5))
    assert np.array_equal(file_values(tools, session, tmp_path / "map.tif"), values(5))


def test_read_virtual_raster(session):
    """A virtual raster of two maps has the values of the maps"""
    tools = Tools(session=session)
    tools.g_region(n=ROWS, s=ROWS // 2, w=0, e=COLS, res=1)
    tools.r_mapcalc(expression="north = row() * 100 + col()")
    tools.g_region(n=ROWS // 2, s=0, w=0, e=COLS, res=1)
    tools.r_mapcalc(expression=f"south = (row() + {ROWS // 2}) * 100 + col()")
    tools.g_region(n=ROWS, s=0, w=0, e=COLS, res=1)
    tools.r_buildvrt(input="north,south", output="mosaic")
    assert np.array_equal(read(session, "mosaic"), values(1))
