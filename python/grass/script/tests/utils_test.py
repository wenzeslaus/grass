"""Test functions in grass.script.utils"""

import os
from pathlib import Path

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
    [
        ("6", 6),
        (" 6 ,2", 6),
        ("", 8),
        ("0", 8),
        ("\u00b2", 8),
        ("9" * 10, 8),
    ],
)
def test_resolve_nprocs_omp_num_threads(monkeypatch, value, expected):
    """OMP_NUM_THREADS is the default only when its first value is a positive number

    A non-ASCII digit (here superscript two) and a value of more than nine
    digits are ignored rather than an error.
    """
    monkeypatch.setattr(gutils, "available_cores", lambda: 8)
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    assert gs.resolve_nprocs(0) == expected
    assert gs.resolve_nprocs(-2) == expected - 2
    assert gs.resolve_nprocs(3) == 3


def test_available_cores():
    """There is at least one core and not more cores than logical CPUs"""
    assert 1 <= gutils.available_cores() <= gs.available_cpus()


def test_available_cores_reads_sysfs():
    """Each distinct list of thread siblings of the allowed CPUs is one core"""
    try:
        cores = {
            Path(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ).read_text(encoding="ascii")
            for cpu in os.sched_getaffinity(0)
        }
    except (AttributeError, OSError):
        pytest.skip("needs CPU affinity and the CPU topology in sysfs (Linux)")
    assert gutils.available_cores() == len(cores)


@pytest.mark.parametrize(
    ("siblings_of_3", "cpus", "expected"), [("2-3", 3, 2), ("2-3", 1, 1), ("", 3, 3)]
)
def test_available_cores_rules(monkeypatch, siblings_of_3, cpus, expected):
    """Siblings count once, at their lowest CPU in the mask (CPU 0 is not in it)

    The result is at most available_cpus() and, with an unreadable topology,
    equal to it.
    """
    topology = {"cpu1": "0,1\n", "cpu2": "2-3\n", "cpu3": siblings_of_3}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 2, 3}, raising=False)
    monkeypatch.setattr(
        Path, "read_text", lambda self: topology[self.parent.parent.name]
    )
    monkeypatch.setattr(gutils, "available_cpus", lambda: cpus)
    assert gutils.available_cores() == expected
