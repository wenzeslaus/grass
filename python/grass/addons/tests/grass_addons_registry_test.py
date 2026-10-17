"""Tests of grass.addons.registry

The golden files in the data directory pin the exact bytes the writers
produce after the bug fixes which followed the extraction from
g.extension. They were generated with the fixed code using the parameter
values recorded in the constants below: the writer golden files
(golden_modules.xml, golden_toolboxes.xml, golden_extensions.xml) used
GOLDEN_PREFIX, and the golden files produced by the installation
functions (golden_extensions_after_install.xml,
golden_modules_after_install.xml) used GOLDEN_RELATIVE_PREFIX with the
current working directory in a temporary directory, so that the paths
embedded in the output are deterministic.

Relative to the pre-extraction g.extension output, the golden files
differ in exactly two ways: correlate entries record the code of the
correlated toolbox instead of the code of the toolbox which contains
them, and empty description and keywords elements are written as empty
elements instead of the string "None".
"""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from grass.addons.registry import LocalRegistry, etree_fromfile

DATA_DIR = Path(__file__).parent / "data"

GOLDEN_PREFIX = "/fixed/addons/prefix"
GOLDEN_RELATIVE_PREFIX = "data/fixed-prefix"
GOLDEN_VERSION_MAJOR = "8"
GOLDEN_LIBGIS_REVISION = "GOLDEN-REVISION"
GOLDEN_EDICT = {
    "v.example.suite": {
        "flist": [
            "bin/v.example.one",
            "scripts/v.example.two",
            "docs/html/v.example.suite.html",
        ],
        "mlist": ["v.example.one", "v.example.two"],
    }
}
GOLDEN_MODULE_LIST = ["i.example.one"]
GOLDEN_MODULE_DESCRIPTION = "Example description"
GOLDEN_MODULE_KEYWORDS = ["example", "test"]

MODULES_FIXTURE_NAMES = [
    "d.frame",
    "d.mon2",
    "g.copyall",
    "g.isis3mt",
    "g.proj.all",
    "r.gdd",
    "r.geomorphon",
    "r.le.patch",
    "r.le.pixel",
    "r.traveltime",
    "r.univar2",
    "v.civil",
    "v.class.ml",
    "v.class.mlpy",
    "v.colors2",
    "v.delaunay3d",
    "v.ellipse",
    "v.in.proj",
    "v.in.redwg",
    "v.neighborhoodmatrix",
    "v.transects",
    "wx.metadata",
]

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Golden files embed the POSIX path separator",
)


class RecordingReporter:
    """Reporter which records received messages for assertions"""

    def __init__(self):
        self.messages = []
        self.verbose_messages = []
        self.debug_messages = []
        self.warnings = []

    def message(self, text):
        self.messages.append(text)

    def verbose(self, text):
        self.verbose_messages.append(text)

    def debug(self, text):
        self.debug_messages.append(text)

    def warning(self, text):
        self.warnings.append(text)

    def check_cancelled(self):
        pass


def make_registry(prefix, reporter=None):
    """Create a registry with the recorded golden parameter values"""
    return LocalRegistry(
        prefix,
        version_major=GOLDEN_VERSION_MAJOR,
        libgis_revision=GOLDEN_LIBGIS_REVISION,
        reporter=reporter,
    )


def make_parse_interface_stub():
    """Create a gtask replacement with the recorded golden interface values"""
    return SimpleNamespace(
        parse_interface=lambda name: SimpleNamespace(
            description=GOLDEN_MODULE_DESCRIPTION, keywords=GOLDEN_MODULE_KEYWORDS
        )
    )


@skip_on_windows
def test_write_xml_modules_matches_golden(tmp_path):
    registry = make_registry(GOLDEN_PREFIX)
    tree = etree_fromfile(DATA_DIR / "modules.xml")
    output = tmp_path / "modules.xml"
    registry.write_xml_modules(output, tree)
    assert output.read_bytes() == (DATA_DIR / "golden_modules.xml").read_bytes()


@skip_on_windows
def test_write_xml_extensions_matches_golden(tmp_path):
    registry = make_registry(GOLDEN_PREFIX)
    tree = etree_fromfile(DATA_DIR / "extensions_input.xml")
    output = tmp_path / "extensions.xml"
    registry.write_xml_extensions(output, tree)
    assert output.read_bytes() == (DATA_DIR / "golden_extensions.xml").read_bytes()


@skip_on_windows
def test_write_xml_toolboxes_matches_golden(tmp_path):
    registry = make_registry(GOLDEN_PREFIX)
    tree = etree_fromfile(DATA_DIR / "toolboxes.xml")
    output = tmp_path / "toolboxes.xml"
    registry.write_xml_toolboxes(output, tree)
    assert output.read_bytes() == (DATA_DIR / "golden_toolboxes.xml").read_bytes()


def test_write_xml_toolboxes_preserves_correlate_codes(tmp_path):
    input_xml = """\
<addons version="8">
    <toolbox name="Hydrology" code="HY">
        <correlate code="RA" />
        <correlate code="VE" />
        <task name="r.stream.basins" />
    </toolbox>
</addons>
"""
    registry = make_registry(str(tmp_path))
    xml_file = tmp_path / "toolboxes.xml"
    registry.write_xml_toolboxes(xml_file, ET.fromstring(input_xml))
    tree = etree_fromfile(xml_file)
    correlates = [
        node.get("code") for node in tree.find("toolbox").findall("correlate")
    ]
    assert correlates == ["RA", "VE"]


def test_write_xml_modules_renders_empty_optional_elements(tmp_path):
    input_xml = """\
<addons version="8">
    <task name="r.example">
        <description></description>
        <keywords></keywords>
    </task>
</addons>
"""
    registry = make_registry(str(tmp_path))
    xml_file = tmp_path / "modules.xml"
    registry.write_xml_modules(xml_file, ET.fromstring(input_xml))
    content = xml_file.read_text(encoding="utf-8")
    assert "None" not in content
    assert "<description></description>" in content
    assert "<keywords></keywords>" in content


@pytest.mark.parametrize("prefix", [GOLDEN_PREFIX, GOLDEN_RELATIVE_PREFIX])
def test_write_xml_modules_rewrite_is_idempotent(tmp_path, prefix):
    # A file whose entries already carry the prefix must not get the
    # prefix prepended again when it is read and written back, for both
    # absolute and relative prefixes.
    input_xml = """\
<addons version="8">
    <task name="x.example">
        <description>Example X</description>
        <keywords>example,x</keywords>
        <binary>
            <file>bin/x.example</file>
            <file>docs/html/x.example.html</file>
        </binary>
    </task>
</addons>
"""
    registry = make_registry(prefix)
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    registry.write_xml_modules(first, ET.fromstring(input_xml))
    registry.write_xml_modules(second, etree_fromfile(first))
    assert first.read_bytes() == second.read_bytes()


def test_write_xml_modules_utf8_round_trip(tmp_path):
    description = "Rastr obsahuje výšku terénu"
    input_xml = f"""\
<addons version="8">
    <task name="r.example">
        <description>{description}</description>
        <keywords>example</keywords>
    </task>
</addons>
"""
    registry = make_registry(str(tmp_path))
    xml_file = tmp_path / "modules.xml"
    registry.write_xml_modules(xml_file, ET.fromstring(input_xml))
    raw = xml_file.read_bytes()
    # The file declares UTF-8, so the bytes must be UTF-8 regardless of
    # the locale, and reading it back must preserve the text.
    assert description.encode("utf-8") in raw
    tree = etree_fromfile(xml_file)
    assert tree.find("task").find("description").text == description


@skip_on_windows
def test_install_extension_xml_matches_golden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = Path(GOLDEN_RELATIVE_PREFIX)
    prefix.mkdir(parents=True)
    registry = make_registry(GOLDEN_RELATIVE_PREFIX)
    registry.install_extension_xml(GOLDEN_EDICT)
    golden = (DATA_DIR / "golden_extensions_after_install.xml").read_bytes()
    assert (prefix / "extensions.xml").read_bytes() == golden


@skip_on_windows
def test_install_module_xml_matches_golden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("grass.addons.registry.gtask", make_parse_interface_stub())
    prefix = Path(GOLDEN_RELATIVE_PREFIX)
    prefix.mkdir(parents=True)
    registry = make_registry(GOLDEN_RELATIVE_PREFIX)
    registry.install_module_xml(GOLDEN_MODULE_LIST)
    golden = (DATA_DIR / "golden_modules_after_install.xml").read_bytes()
    assert (prefix / "modules.xml").read_bytes() == golden


def test_get_installed_modules_names(tmp_path):
    shutil.copy(DATA_DIR / "modules.xml", tmp_path / "modules.xml")
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_modules() == MODULES_FIXTURE_NAMES


@skip_on_windows
def test_get_installed_modules_shell_format(tmp_path):
    # Executables are only recognized under prefix/bin and prefix/scripts,
    # and the file paths in the written file have the prefix prepended, so
    # write the file through the registry from a tree with relative paths.
    input_xml = """\
<addons version="8">
    <task name="x.example">
        <description>Example X</description>
        <keywords>example,x</keywords>
        <binary>
            <file>bin/x.example</file>
            <file>docs/html/x.example.html</file>
        </binary>
    </task>
    <task name="y.example">
        <description>Example Y</description>
        <keywords>example,y</keywords>
        <binary>
            <file>scripts/y.example</file>
        </binary>
    </task>
</addons>
"""
    registry = make_registry(str(tmp_path))
    registry.write_xml_modules(tmp_path / "modules.xml", ET.fromstring(input_xml))
    assert registry.get_installed_modules(shell_format=True) == [
        "name=x.example",
        "description=Example X",
        "keywords=example,x",
        "executables=x.example",
        "name=y.example",
        "description=Example Y",
        "keywords=example,y",
        "executables=y.example",
    ]


def test_get_installed_modules_missing_file_without_force(tmp_path):
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_modules() == []
    assert not (tmp_path / "modules.xml").exists()


def test_get_installed_modules_missing_file_with_force(tmp_path):
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_modules(force=True) == []
    xml_file = tmp_path / "modules.xml"
    assert xml_file.exists()
    assert ET.fromstring(xml_file.read_text()).findall("task") == []


def test_get_installed_modules_corrupt_file(tmp_path):
    xml_file = tmp_path / "modules.xml"
    xml_file.write_text("not really XML")
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_modules() == []
    assert ET.fromstring(xml_file.read_text()).findall("task") == []


def test_get_installed_toolboxes_codes(tmp_path):
    shutil.copy(DATA_DIR / "toolboxes.xml", tmp_path / "toolboxes.xml")
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_toolboxes() == ["HY", "MC"]


def test_get_installed_toolboxes_missing_file_without_force(tmp_path):
    reporter = RecordingReporter()
    registry = make_registry(str(tmp_path), reporter=reporter)
    assert registry.get_installed_toolboxes() == []
    assert not (tmp_path / "toolboxes.xml").exists()
    assert reporter.debug_messages == ["No addons metadata file available"]


def test_get_installed_toolboxes_missing_file_with_force(tmp_path):
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_toolboxes(force=True) == []
    xml_file = tmp_path / "toolboxes.xml"
    assert xml_file.exists()
    assert ET.fromstring(xml_file.read_text()).findall("toolbox") == []


def test_get_installed_toolboxes_corrupt_file(tmp_path):
    xml_file = tmp_path / "toolboxes.xml"
    xml_file.write_text("not really XML")
    registry = make_registry(str(tmp_path))
    assert registry.get_installed_toolboxes() == []
    assert ET.fromstring(xml_file.read_text()).findall("toolbox") == []


def test_install_toolbox_xml_creates_toolbox(tmp_path):
    registry = make_registry(str(tmp_path))
    tdata = {
        "name": "Hydrology tools",
        "correlate": ["RA"],
        "modules": ["r.one", "r.two"],
    }
    registry.install_toolbox_xml("HY", tdata)
    tree = ET.fromstring((tmp_path / "toolboxes.xml").read_text())
    toolboxes = tree.findall("toolbox")
    assert len(toolboxes) == 1
    assert toolboxes[0].get("code") == "HY"
    assert toolboxes[0].get("name") == "Hydrology tools"
    tasks = [node.get("name") for node in toolboxes[0].findall("task")]
    assert tasks == ["r.one", "r.two"]
    correlates = [node.get("code") for node in toolboxes[0].findall("correlate")]
    assert correlates == ["RA"]


def test_install_toolbox_xml_missing_file_gets_toolbox_doctype(tmp_path):
    registry = make_registry(str(tmp_path))
    tdata = {"name": "Hydrology tools", "correlate": [], "modules": []}
    registry.install_toolbox_xml("HY", tdata)
    lines = (tmp_path / "toolboxes.xml").read_text(encoding="utf-8").splitlines()
    assert lines[1] == '<!DOCTYPE toolbox SYSTEM "grass-addons.dtd">'


def test_install_toolbox_xml_updates_without_duplicating(tmp_path):
    registry = make_registry(str(tmp_path))
    tdata = {
        "name": "Hydrology tools",
        "correlate": ["RA"],
        "modules": ["r.one", "r.two"],
    }
    registry.install_toolbox_xml("HY", tdata)
    updated = {
        "name": "Hydrology tools",
        "correlate": ["RA"],
        "modules": ["r.three"],
    }
    registry.install_toolbox_xml("HY", updated)
    tree = ET.fromstring((tmp_path / "toolboxes.xml").read_text())
    toolboxes = tree.findall("toolbox")
    assert len(toolboxes) == 1
    tasks = [node.get("name") for node in toolboxes[0].findall("task")]
    assert tasks == ["r.three"]


def test_remove_extension_xml_removes_entries(tmp_path, monkeypatch):
    monkeypatch.setattr("grass.addons.registry.gtask", make_parse_interface_stub())
    registry = make_registry(str(tmp_path))
    edict = {
        "r.example.single": {
            "flist": ["bin/r.example.single"],
            "mlist": ["r.example.single"],
        }
    }
    registry.install_module_xml(["r.example.single"])
    registry.install_extension_xml(edict)
    assert registry.get_installed_modules() == ["r.example.single"]
    registry.remove_extension_xml(["r.example.single"], edict, "r.example.single")
    assert registry.get_installed_modules() == []
    tree = ET.fromstring((tmp_path / "extensions.xml").read_text())
    assert tree.findall("task") == []


def test_remove_extension_xml_tolerates_missing_files(tmp_path):
    registry = make_registry(str(tmp_path))
    # More than one entry in edict also exercises the toolboxes file update.
    edict = {
        "r.example.one": {"flist": [], "mlist": []},
        "r.example.two": {"flist": [], "mlist": []},
    }
    registry.remove_extension_xml(["r.example.one", "r.example.two"], edict, "example")
    assert list(tmp_path.iterdir()) == []


def test_remove_from_toolbox_xml_removes_toolbox(tmp_path):
    shutil.copy(DATA_DIR / "toolboxes.xml", tmp_path / "toolboxes.xml")
    registry = make_registry(str(tmp_path))
    registry.remove_from_toolbox_xml("HY")
    assert registry.get_installed_toolboxes() == ["MC"]


def test_remove_from_toolbox_xml_tolerates_missing_file(tmp_path):
    registry = make_registry(str(tmp_path))
    registry.remove_from_toolbox_xml("HY")
    assert not (tmp_path / "toolboxes.xml").exists()


def test_reporter_receives_debug_for_missing_modules_file(tmp_path):
    reporter = RecordingReporter()
    registry = make_registry(str(tmp_path), reporter=reporter)
    registry.get_installed_modules()
    assert reporter.debug_messages == ["No addons metadata file available"]


def test_reporter_receives_warning_for_failing_interface(tmp_path, monkeypatch):
    def parse_interface(name):
        message = "no interface"
        raise RuntimeError(message)

    monkeypatch.setattr(
        "grass.addons.registry.gtask",
        SimpleNamespace(parse_interface=parse_interface),
    )
    reporter = RecordingReporter()
    registry = make_registry(str(tmp_path), reporter=reporter)
    registry.install_module_xml(["r.example.single"])
    assert len(reporter.warnings) == 1
    assert "r.example.single" in reporter.warnings[0]
    assert "no interface" in reporter.warnings[0]


def test_reporter_receives_verbose_for_existing_module(tmp_path, monkeypatch):
    monkeypatch.setattr("grass.addons.registry.gtask", make_parse_interface_stub())
    reporter = RecordingReporter()
    registry = make_registry(str(tmp_path), reporter=reporter)
    registry.install_module_xml(["r.example.single"])
    assert reporter.verbose_messages == []
    registry.install_module_xml(["r.example.single"])
    assert reporter.verbose_messages == [
        "Extension module already listed in metadata file; metadata not updated!"
    ]


def test_reporter_receives_verbose_for_existing_extension(tmp_path):
    reporter = RecordingReporter()
    registry = make_registry(str(tmp_path), reporter=reporter)
    edict = {
        "r.example.single": {
            "flist": ["bin/r.example.single"],
            "mlist": ["r.example.single"],
        }
    }
    registry.install_extension_xml(edict)
    assert reporter.verbose_messages == []
    registry.install_extension_xml(edict)
    assert reporter.verbose_messages == [
        "Extension already listed in metadata file; metadata not updated!"
    ]
