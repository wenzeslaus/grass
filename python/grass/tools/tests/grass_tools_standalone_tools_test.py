"""Test grass.tools.StandaloneTools class"""

from pathlib import Path

import pytest

from grass.experimental import MapsetSession
from grass.tools import StandaloneTools, Tools


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own directory because outputs are relative paths"""
    monkeypatch.chdir(tmp_path)


def test_json():
    """Check that numbers are parsed as numbers"""
    tools = StandaloneTools()
    assert tools.run_from_list(["g.region", "-p", "format=json"])["nsres"] == 1


def test_run_name_and_parameters():
    """Check that name-and-parameters style works"""
    tools = StandaloneTools()
    assert tools.run("g.region", flags="p", format="json")["nsres"] == 1


# TODO: This may fail because shutils.which does not setup a session.
def test_function_call_with_json():
    """Check that numbers are parsed as numbers"""
    tools = StandaloneTools()
    assert tools.g_region(flags="p", format="json")["nsres"] == 1


def test_key_value_parser_number():
    """Check that numbers are parsed as numbers"""
    tools = StandaloneTools()
    assert tools.run_from_list(["g.region", "-g"]).keyval["nsres"] == 1


def test_pack_input_output(rows_raster_file3x2):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
    assert Path(rows_raster_file3x2).exists()
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


def test_pack_initial_region_set(rows_raster_file3x2, rows_raster_file4x5):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
    assert Path(rows_raster_file3x2).exists()
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    region = tools.g_region(flags="p", format="json")
    assert region["rows"] == 3
    assert region["cols"] == 2


def test_pack_initial_region_respected(rows_raster_file3x2, rows_raster_file4x5):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
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
    region = tools.r_info(map="file.grass_raster", format="json")
    assert region["rows"] == 3
    assert region["cols"] == 2
    region = tools.r_info(map="file2.grass_raster", format="json")
    assert region["rows"] == 3
    assert region["cols"] == 2


def test_pack_manual_region_from_raster_preserved(
    rows_raster_file3x2, rows_raster_file4x5
):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
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


def test_pack_manual_region_from_parameters_preserved(rows_raster_file3x2):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
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


def test_pack_manual_region_read_but_not_set(rows_raster_file3x2):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
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


def test_pack_region_manually_changed(rows_raster_file3x2, rows_raster_file4x5):
    """Check that global overwrite is not used when separate env is used"""
    tools = StandaloneTools()
    tools.r_slope_aspect(
        elevation=rows_raster_file3x2,
        aspect="file.grass_raster",
        flags="a",
    )
    region = tools.g_region(rows=2, cols=2)
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


def test_pack_workflow(rows_raster_file4x5):
    tools = StandaloneTools()
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
    assert not Path("elevation.grr").exists()


def test_regions_independent():
    """Check that global overwrite is not used when separate env is used"""
    tools1 = StandaloneTools()
    tools1.g_region(rows=2, cols=3)
    tools2 = StandaloneTools()
    tools2.g_region(rows=4, cols=5)
    region = tools1.g_region(flags="p", format="json")
    assert region["rows"] == 2
    assert region["cols"] == 3
    region = tools2.g_region(flags="p", format="json")
    assert region["rows"] == 4
    assert region["cols"] == 5


def test_region_changed_for_session(xy_dataset_session):
    """Check that global overwrite is not used when separate env is used"""
    tools1 = StandaloneTools(session=xy_dataset_session)
    tools1.g_region(rows=4, cols=5)
    tools2 = StandaloneTools(session=xy_dataset_session)
    region = tools2.g_region(flags="p", format="json")
    assert region["rows"] == 4
    assert region["cols"] == 5


# May fail depending on the order of the tests because of the session setup.
def test_existing_session_accepted(xy_dataset_session):
    """Check that numbers are parsed as numbers"""
    with MapsetSession("test1", create=True, env=xy_dataset_session.env) as session:
        tools = StandaloneTools(session=session)
        assert tools.g_mapset(flags="p").text == session.name


def test_another_session_is_ignored(xy_dataset_session):
    """Check that numbers are parsed as numbers"""
    with MapsetSession("test1", create=True, env=xy_dataset_session.env) as session:
        tools = StandaloneTools()
        assert tools.g_mapset(flags="p").text == "PERMANENT"
        tools = Tools(session=session)
        assert tools.g_mapset(flags="p").text == "test1"
