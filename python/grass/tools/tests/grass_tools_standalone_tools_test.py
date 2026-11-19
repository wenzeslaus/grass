# SPDX-FileCopyrightText: 2025-2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
"""Test grass.tools.StandaloneTools class

The region tests pin the current behavior, which is not a documented contract.
"""

from pathlib import Path

import pytest

from grass.experimental import MapsetSession
from grass.tools import StandaloneTools, ToolError


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own directory because outputs are relative paths"""
    monkeypatch.chdir(tmp_path)


def test_json(tmp_path):
    """Check that JSON output of a tool run from a list of strings is parsed"""
    tools = StandaloneTools(work_dir=tmp_path)
    assert tools.run_from_list(["g.region", "-p", "format=json"])["nsres"] == 1


def test_run_name_and_parameters(tmp_path):
    """Check that a tool runs when given as a name and parameters"""
    tools = StandaloneTools(work_dir=tmp_path)
    assert tools.run("g.region", flags="p", format="json")["nsres"] == 1


def test_function_call_with_json():
    """Check that a tool runs as a function named after it without a work directory"""
    tools = StandaloneTools()
    assert tools.g_region(flags="p", format="json")["nsres"] == 1


def test_key_value_parser_number(tmp_path):
    """Check that key-value output has numbers parsed as numbers"""
    tools = StandaloneTools(work_dir=tmp_path)
    assert tools.run_from_list(["g.region", "-g"]).keyval["nsres"] == 1


def test_pack_input_output(rows_raster_file3x2, tmp_path):
    """Check that an output file is written from an input file, also when it exists"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.run_from_list(
        [
            "r.slope.aspect",
            f"elevation={rows_raster_file3x2}",
            "aspect=file.grass_raster",
        ]
    )
    tools.run_from_list(
        [
            "r.slope.aspect",
            f"elevation={rows_raster_file3x2}",
            "aspect=file.grass_raster",
        ]
    )
    assert Path("file.grass_raster").exists()


def test_pack_initial_region_set(rows_raster_file3x2, tmp_path):
    """Check that the region is set from the first input raster file"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 3
    assert region["cols"] == 2


def test_pack_initial_region_respected(
    rows_raster_file3x2, rows_raster_file4x5, tmp_path
):
    """Check that the region from the first input file is kept for a later input"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    tools.r_slope_aspect(
        elevation=rows_raster_file4x5,
        aspect="file2.grass_raster",
        flags="a",
    )
    assert Path("file.grass_raster").exists()
    assert Path("file2.grass_raster").exists()
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 3
    assert region["cols"] == 2
    info = tools.r_info(map="file.grass_raster", format="json")
    assert info["rows"] == 3
    assert info["cols"] == 2
    info = tools.r_info(map="file2.grass_raster", format="json")
    assert info["rows"] == 3
    assert info["cols"] == 2


def test_pack_manual_region_from_raster_preserved(
    rows_raster_file3x2, rows_raster_file4x5, tmp_path
):
    """Check that a region set with g.region from a raster file is kept"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.g_region(raster=rows_raster_file4x5)
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    assert Path("file.grass_raster").exists()
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 4
    assert region["cols"] == 5


def test_pack_manual_region_from_parameters_preserved(rows_raster_file3x2, tmp_path):
    """Check that a region set with g.region before the first input file is kept"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.g_region(rows=4, cols=5)
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    assert Path("file.grass_raster").exists()
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 4
    assert region["cols"] == 5


def test_pack_manual_region_read_but_not_set(rows_raster_file3x2, tmp_path):
    """Check that printing the region does not count as setting it"""
    tools = StandaloneTools(work_dir=tmp_path)
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 1
    assert region["cols"] == 1
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    assert Path("file.grass_raster").exists()
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 3
    assert region["cols"] == 2


def test_pack_region_manually_changed(
    rows_raster_file3x2, rows_raster_file4x5, tmp_path
):
    """Check that a region set with g.region after the first input file is kept"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    tools.g_region(rows=2, cols=2)
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 2
    tools.r_slope_aspect(
        elevation=rows_raster_file4x5,
        aspect="file2.grass_raster",
        flags="a",
    )
    assert Path("file.grass_raster").exists()
    assert Path("file2.grass_raster").exists()
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 2


def test_pack_crs_from_first_input(ones_raster_file_epsg3358, tmp_path):
    """Check that the CRS and the region come from the first input raster"""
    tools = StandaloneTools(work_dir=tmp_path)
    assert tools.r_univar(map=ones_raster_file_epsg3358, format="json")["n"] == 20
    assert tools.g_proj(flags="p", format="projjson")["id"]["code"] == 3358


def test_pack_crs_region_set_before_first_input_preserved(
    ones_raster_file_epsg3358, tmp_path
):
    """Check that a region set before the CRS is known is kept"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.g_region(n=20, s=0, e=30, w=0, res=10)
    assert tools.r_univar(map=ones_raster_file_epsg3358, format="json")["cells"] == 6
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 3


def test_pack_crs_mismatch(
    ones_raster_file_epsg3358, ones_raster_file_epsg4326, tmp_path
):
    """Check that an input with a CRS different from the first one is an error"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.r_univar(map=ones_raster_file_epsg3358, format="json")
    with pytest.raises(ToolError, match="does not appear to match"):
        tools.r_univar(map=ones_raster_file_epsg4326, format="json")


def test_pack_workflow(rows_raster_file4x5, tmp_path):
    """Check that rasters in the project are used by name and only files are written"""
    tools = StandaloneTools(work_dir=tmp_path)
    tools.r_slope_aspect(
        elevation=rows_raster_file4x5, slope="slope.grr", aspect="aspect"
    )
    tools.r_flow(
        elevation=rows_raster_file4x5.stem,
        aspect="aspect",
        flowaccumulation="flow_accumulation.grr",
        flags=3,
    )
    assert Path(rows_raster_file4x5).exists()
    assert Path("slope.grr").exists()
    assert Path("flow_accumulation.grr").exists()
    assert not Path("aspect.grr").exists()


def test_regions_independent(tmp_path):
    """Check that two objects have separate regions"""
    tools1 = StandaloneTools(work_dir=tmp_path / "first")
    tools1.g_region(rows=2, cols=3)
    tools2 = StandaloneTools(work_dir=tmp_path / "second")
    tools2.g_region(rows=4, cols=5)
    region = tools1.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 3
    region = tools2.g_region(flags="p", format="json")
    assert region["rows"] == 4
    assert region["cols"] == 5


def test_region_changed_for_session(xy_dataset_session):
    """Check that a region set by one object is seen by another with the same session"""
    tools1 = StandaloneTools(session=xy_dataset_session)
    tools1.g_region(rows=4, cols=5)
    tools2 = StandaloneTools(session=xy_dataset_session)
    region = tools2.g_region(flags="p", format="json")
    assert region["rows"] == 4
    assert region["cols"] == 5


def test_existing_session_accepted(xy_dataset_session):
    """Check that tools run in the mapset of a given session"""
    with MapsetSession("test1", create=True, env=xy_dataset_session.env) as session:
        tools = StandaloneTools(session=session)
        assert tools.g_mapset(flags="p").text == session.name


def test_another_session_is_ignored(xy_dataset_session, tmp_path):
    """Check that an object without a given session runs in its own project"""
    with MapsetSession("test1", create=True, env=xy_dataset_session.env):
        tools = StandaloneTools(work_dir=tmp_path)
        assert tools.g_mapset(flags="p").text == "PERMANENT"
