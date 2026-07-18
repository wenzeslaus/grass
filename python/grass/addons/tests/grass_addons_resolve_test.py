"""Tests of grass.addons.resolve

All tests run without a GRASS session and without network access; network
and git interactions are replaced by monkeypatching attributes on the
grass.addons.resolve module.
"""

import sys
import types
import urllib.error

import pytest

from grass.addons.config import OFFICIAL_REPOSITORY_URL
from grass.addons.exceptions import SourceResolutionError
from grass.addons.resolve import (
    SUPPORTED_TAR_FORMATS,
    get_default_branch,
    get_version_branch,
    resolve_known_host_service,
    resolve_source_code,
    resolve_xmlurl_prefix,
    validate_url,
)


@pytest.fixture
def no_network(monkeypatch):
    """Make remote-URL resolution work without any network access"""
    monkeypatch.setattr("grass.addons.resolve.validate_url", lambda url: True)
    monkeypatch.setattr("grass.addons.resolve.get_default_branch", lambda url: "main")


def make_gs_stub(out=b"", err=b""):
    """Create a stand-in for the grass.script namespace used by resolve

    The returned namespace provides Popen and decode, with Popen producing
    the given bytes as the stdout and stderr of git ls-remote.
    """

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None):
            self.cmd = cmd

        def communicate(self):
            return out, err

    return types.SimpleNamespace(Popen=FakePopen, decode=lambda data: data.decode())


def test_resolve_source_code_official_without_url():
    assert resolve_source_code() == ("official", OFFICIAL_REPOSITORY_URL)


def test_resolve_source_code_official_with_url():
    assert resolve_source_code(url=OFFICIAL_REPOSITORY_URL) == (
        "official",
        OFFICIAL_REPOSITORY_URL,
    )


def test_resolve_source_code_local_directory(tmp_path):
    assert resolve_source_code(url=str(tmp_path)) == ("dir", str(tmp_path))


def test_resolve_source_code_local_zip_file(tmp_path):
    zip_file = tmp_path / "g.example.zip"
    zip_file.write_bytes(b"")
    assert resolve_source_code(url=str(zip_file)) == ("zip", str(zip_file))


@pytest.mark.parametrize("suffix", SUPPORTED_TAR_FORMATS)
def test_resolve_source_code_local_tar_file(tmp_path, suffix):
    archive = tmp_path / f"g.example.{suffix}"
    archive.write_bytes(b"")
    assert resolve_source_code(url=str(archive)) == (suffix, str(archive))


def test_resolve_source_code_file_url_prefix(tmp_path):
    assert resolve_source_code(url=f"file://{tmp_path}") == ("dir", str(tmp_path))


def test_resolve_source_code_fork_local_directory(tmp_path):
    assert resolve_source_code(url=str(tmp_path), fork=True) == (
        "official_fork",
        str(tmp_path),
    )


@pytest.mark.parametrize(
    "url",
    [
        "github.com/user/g.example",
        "https://github.com/user/g.example",
        "https://github.com/user/g.example/",
    ],
)
@pytest.mark.parametrize("branch", ["main", None])
def test_resolve_source_code_github(no_network, url, branch):
    assert resolve_source_code(url=url, name="g.example", branch=branch) == (
        "remote_zip",
        "https://github.com/user/g.example/archive/main.zip",
    )


@pytest.mark.parametrize(
    "url",
    [
        "gitlab.com/JoeUser/GrassModule",
        "https://gitlab.com/JoeUser/GrassModule",
    ],
)
def test_resolve_source_code_gitlab(no_network, url):
    assert resolve_source_code(url=url, name="GrassModule") == (
        "remote_zip",
        "https://gitlab.com/JoeUser/GrassModule/-/archive/main/GrassModule-main.zip",
    )


@pytest.mark.parametrize(
    "url",
    [
        "bitbucket.org/joe-user/grass-module",
        "https://bitbucket.org/joe-user/grass-module",
    ],
)
def test_resolve_source_code_bitbucket(no_network, url):
    assert resolve_source_code(url=url, name="grass-module") == (
        "remote_zip",
        "https://bitbucket.org/joe-user/grass-module/get/main.zip",
    )


def test_resolve_source_code_trac_zip_suffix_unchanged(no_network):
    # A URL already ending with the service's download suffix is not
    # rewritten for the known host, but it still resolves as a remote ZIP.
    url = "https://trac.osgeo.org/grass/browser/r.example?format=zip"
    assert resolve_source_code(url=url, name="r.example") == ("remote_zip", url)


def test_resolve_source_code_github_zip_suffix_unchanged(no_network):
    url = "https://github.com/user/g.example/archive/main.zip"
    assert resolve_source_code(url=url, name="g.example") == ("remote_zip", url)


def test_resolve_source_code_unknown_host_plain_url(no_network):
    # An unrecognized remote URL resolves to the plain "url" type which is
    # usable as a metadata location for listing addons.
    url = "https://example.com/user/g.example"
    assert resolve_source_code(url=url, name="g.example") == ("url", url)


def test_resolve_source_code_local_unsupported_archive(tmp_path):
    archive = tmp_path / "foo.rar"
    archive.write_bytes(b"")
    with pytest.raises(SourceResolutionError, match=r"foo\.rar"):
        resolve_source_code(url=str(archive))


def test_resolve_source_code_unknown_host_tar_gz(no_network):
    url = "https://example.com/user/g.example.tar.gz"
    assert resolve_source_code(url=url, name="g.example") == ("remote_tar.gz", url)


def test_resolve_known_host_service_unknown_domain():
    assert resolve_known_host_service(
        "https://example.com/user/g.example", "g.example", None
    ) == (None, None)


def test_resolve_xmlurl_prefix_custom_url():
    assert (
        resolve_xmlurl_prefix("https://example.com/addons", major_version=8)
        == "https://example.com/addons/"
    )


def test_resolve_xmlurl_prefix_custom_url_trailing_slash():
    assert (
        resolve_xmlurl_prefix("https://example.com/addons/", major_version=8)
        == "https://example.com/addons/"
    )


def test_resolve_xmlurl_prefix_official(monkeypatch):
    monkeypatch.setattr(
        "grass.addons.resolve.get_version_branch",
        lambda major_version: f"grass{major_version}",
    )
    assert (
        resolve_xmlurl_prefix("https://example.com/addons", "official", major_version=8)
        == "https://grass.osgeo.org/addons/grass8/"
    )


def test_validate_url_existing_local_path(tmp_path):
    assert validate_url(str(tmp_path)) is True


def test_validate_url_unreachable(monkeypatch):
    def raising_urlopen(url, *args, **kwargs):
        reason = "connection refused"
        raise urllib.error.URLError(reason)

    monkeypatch.setattr("grass.addons.resolve._urlopen", raising_urlopen)
    with pytest.raises(SourceResolutionError, match="Cannot open URL"):
        validate_url("http://example.com/g.example")


def test_get_version_branch_existing(monkeypatch):
    monkeypatch.setattr(
        "grass.addons.resolve.gs",
        make_gs_stub(out=b"41400fcbba\trefs/heads/grass8\n"),
    )
    assert get_version_branch(8) == "grass8"


def test_get_version_branch_missing(monkeypatch):
    monkeypatch.setattr("grass.addons.resolve.gs", make_gs_stub(out=b""))
    assert get_version_branch(8) == "grass7"


def test_get_version_branch_git_error(monkeypatch):
    monkeypatch.setattr(
        "grass.addons.resolve.gs",
        make_gs_stub(err=b"fatal: unable to access repository"),
    )
    with pytest.raises(SourceResolutionError, match="Failed to get branch"):
        get_version_branch(8)


def test_get_version_branch_win32(monkeypatch):
    # On Windows, the current version branch is assumed without querying git.
    monkeypatch.setattr(sys, "platform", "win32")
    assert get_version_branch(8) == "grass8"


def test_get_default_branch_api_unreachable(monkeypatch):
    def raising_urlopen(url):
        reason = "no network"
        raise urllib.error.URLError(reason)

    monkeypatch.setattr(
        "grass.addons.resolve.urlrequest",
        types.SimpleNamespace(urlopen=raising_urlopen),
    )
    assert get_default_branch("https://github.com/user/g.example") == "main"


def test_get_default_branch_unknown_host():
    # An unknown hosting service returns "main" without any network request,
    # so no monkeypatching of urlopen is needed here.
    assert get_default_branch("https://example.com/user/g.example") == "main"


def test_get_default_branch_without_organization_and_repository():
    # The URL is rejected before any network request is attempted.
    with pytest.raises(
        SourceResolutionError, match="Cannot retrieve organization and repository"
    ):
        get_default_branch("https://github.com")


def test_get_default_branch_github(monkeypatch):
    class FakeResponse:
        def read(self):
            return b'{"default_branch": "dev"}'

    monkeypatch.setattr(
        "grass.addons.resolve.urlrequest",
        types.SimpleNamespace(urlopen=lambda url: FakeResponse()),
    )
    assert get_default_branch("https://github.com/user/g.example") == "dev"
