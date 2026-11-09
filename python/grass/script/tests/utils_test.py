"""Test functions in grass.script.utils"""

import pytest

import grass.script as gs
from grass.script import utils as gutils


def test_named_separators():
    """Check that named separators are recognized and correctly evaluated"""
    assert gs.separator("pipe") == "|"
    assert gs.separator("comma") == ","
    assert gs.separator("space") == " "
    assert gs.separator("tab") == "\t"
    assert gs.separator("newline") == "\n"


def test_backslash_separators():
    """Check that separators specified as an escape sequence are correctly evaluated"""
    assert gs.separator(r"\t") == "\t"
    assert gs.separator(r"\n") == "\n"


def test_unrecognized_separator():
    """Check that unknown strings are just passed through"""
    assert gs.separator("apple") == "apple"


def test_KeyValue_keys():
    """Check that KeyValue class works like a Dict"""
    kv = gs.KeyValue()
    kv["key1"] = "value1"
    kv["key2"] = "value2"

    assert kv["key1"] == "value1"
    assert kv["key2"] == "value2"

    # Raises KeyError for non-existing keys
    with pytest.raises(KeyError):
        _ = kv["non_existing_key"]


def test_KeyValue_values():
    """Check that keys of KeyValue class can be accessed as attributes"""
    kv = gs.KeyValue()
    kv["key1"] = "value1"
    kv["key2"] = "value2"

    assert kv.key1 == "value1"
    assert kv.key2 == "value2"

    # Raises AttributeError for non-existing attribute
    with pytest.raises(AttributeError):
        _ = kv.non_existing_attribute


def test_resolve_nprocs(monkeypatch):
    """Check G_OPT_M_NPROCS resolution: positive as-is, 0 = cores, negative reserves."""
    monkeypatch.setattr(gutils, "available_cores", lambda: 8)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert gs.resolve_nprocs(1) == 1
    assert gs.resolve_nprocs(4) == 4
    assert gs.resolve_nprocs("3") == 3
    assert gs.resolve_nprocs(0) == 8
    assert gs.resolve_nprocs(-2) == 6
    assert gs.resolve_nprocs(-10) == 1
    with pytest.raises(ValueError, match="invalid literal for int"):
        gs.resolve_nprocs("not-a-number")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("6", 6), ("6,2", 6), ("", 8), ("0", 8), ("x", 8), ("\u00b2", 8)],
)
def test_resolve_nprocs_omp_num_threads(monkeypatch, value, expected):
    """OMP_NUM_THREADS replaces the cores only when its first value is a number

    A non-ASCII digit (here superscript two) is not a number, like in C.
    """
    monkeypatch.setattr(gutils, "available_cores", lambda: 8)
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    assert gs.resolve_nprocs(0) == expected
    assert gs.resolve_nprocs(-2) == expected - 2
    assert gs.resolve_nprocs(3) == 3


def test_available_cores():
    """There is at least one core and not more cores than logical CPUs"""
    assert 1 <= gs.available_cores() <= gs.available_cpus()
