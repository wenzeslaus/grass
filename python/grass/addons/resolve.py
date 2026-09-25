# MODULE:    grass.addons
#
# AUTHOR(S): Markus Neteler
#            Martin Landa <landa.martin gmail com>
#            Vaclav Petras <wenzeslaus gmail com>
#
# PURPOSE:   Addon (extension) management library
#
# COPYRIGHT: (C) 2009-2026 by Markus Neteler, and the GRASS Development Team
#
#            This program is free software under the GNU General Public
#            License (>=v2). Read the file COPYING that comes with GRASS
#            for details.

"""Resolution of addon source code locations (names and URLs to source types)"""

import json
import os
import sys
from pathlib import Path
from subprocess import PIPE
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlparse

import grass.script as gs

from .config import HTTP_HEADERS, OFFICIAL_REPOSITORY_URL
from .exceptions import SourceResolutionError
from .reporter import NullReporter

# This duplicates the list set as extract_tar.supported_formats in g.extension
# until the download code is consolidated in this package in a later stage.
SUPPORTED_TAR_FORMATS = ("tar.gz", "gz", "bz2", "tar", "gzip", "targz")


def _urlopen(url, *args, **kwargs):
    """Wrapper around urlopen. Same function as 'urlopen', but with the
    ability to define headers.
    """
    request = urlrequest.Request(url, headers=HTTP_HEADERS)
    return urlrequest.urlopen(request, *args, **kwargs)


def get_version_branch(major_version):
    """Check if version branch for the current GRASS version exists,
    if not, take branch for the previous version
    For the official repo we assume that at least one version branch is present"""
    version_branch = f"grass{major_version}"
    if sys.platform == "win32":
        return version_branch
    branch = gs.Popen(
        [
            "git",
            "ls-remote",
            "--heads",
            OFFICIAL_REPOSITORY_URL,
            f"refs/heads/{version_branch}",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    branch, stderr = branch.communicate()
    if stderr:
        raise SourceResolutionError(
            _(
                "Failed to get branch from the Git repository <{repo_path}>.\n{error}"
            ).format(
                repo_path=OFFICIAL_REPOSITORY_URL,
                error=gs.decode(stderr),
            )
        )
    branch = gs.decode(branch)
    if version_branch not in branch:
        version_branch = "grass{}".format(int(major_version) - 1)
    return version_branch


def get_default_branch(full_url):
    """Get default branch for repository in known hosting services
    (currently only implemented for github, gitlab and bitbucket API)
    In all other cases "main" is used as default"""
    # Parse URL
    url_parts = urlparse(full_url)
    # Get organization and repository component
    try:
        organization, repository = url_parts.path.split("/")[1:3]
    except ValueError as error:
        raise SourceResolutionError(
            _("Cannot retrieve organization and repository from URL: <{}>.").format(
                full_url
            )
        ) from error
    # Construct API call and retrieve default branch
    api_calls = {
        "github.com": f"https://api.github.com/repos/{organization}/{repository}",
        "gitlab.com": f"https://gitlab.com/api/v4/projects/{organization}%2F{repository}",  # noqa: E501
        "bitbucket.org": f"https://api.bitbucket.org/2.0/repositories/{organization}/{repository}/branching-model?",  # noqa: E501
    }
    api_call = api_calls.get(url_parts.netloc)
    if api_call is None:
        return "main"
    # Try to get default branch via API. The API call is known to fail
    # if the rate limit of the API is exceeded.
    try:
        req = urlrequest.urlopen(api_call)
        content = json.loads(req.read())
        # For github and gitlab
        default_branch = content.get("default_branch")
        # For bitbucket
        if not default_branch:
            default_branch = content.get("development").get("name")
    except URLError:
        default_branch = "main"
    return default_branch


def resolve_xmlurl_prefix(url, source=None, *, major_version, reporter=None):
    """Determine and check the URL where the XML metadata files are stored

    It ensures that there is a single slash at the end of URL, so we can attach
     file name easily:

    >>> resolve_xmlurl_prefix("https://grass.osgeo.org/addons", major_version=8)
    'https://grass.osgeo.org/addons/'
    >>> resolve_xmlurl_prefix("https://grass.osgeo.org/addons/", major_version=8)
    'https://grass.osgeo.org/addons/'
    """
    if reporter is None:
        reporter = NullReporter()
    reporter.debug("resolve_xmlurl_prefix(url={0}, source={1})".format(url, source))
    if source in {"official", "official_fork"}:
        # use pregenerated modules XML file
        # Define branch to fetch from (latest or current version)
        version_branch = get_version_branch(major_version)

        url = "https://grass.osgeo.org/addons/{}/".format(version_branch)
    # else try to get extensions XMl from SVN repository (provided URL)
    # the exact action depends on subsequent code (somewhere)

    if not url.endswith("/"):
        url += "/"
    return url


KNOWN_HOST_SERVICES_INFO = {
    "OSGeo Trac": {
        "domain": "trac.osgeo.org",
        "ignored_suffixes": ["format=zip"],
        "possible_starts": ["", "https://", "http://"],
        "url_start": "https://",
        "url_end": "?format=zip",
    },
    "GitHub": {
        "domain": "github.com",
        "ignored_suffixes": [".zip", ".tar.gz"],
        "possible_starts": ["", "https://", "http://"],
        "url_start": "https://",
        "url_end": "/archive/{branch}.zip",
    },
    "GitLab": {
        "domain": "gitlab.com",
        "ignored_suffixes": [".zip", ".tar.gz", ".tar.bz2", ".tar"],
        "possible_starts": ["", "https://", "http://"],
        "url_start": "https://",
        "url_end": "/-/archive/{branch}/{name}-{branch}.zip",
    },
    "Bitbucket": {
        "domain": "bitbucket.org",
        "ignored_suffixes": [".zip", ".tar.gz", ".gz", ".bz2"],
        "possible_starts": ["", "https://", "http://"],
        "url_start": "https://",
        "url_end": "/get/{branch}.zip",
    },
}

# TODO: support ZIP URLs which don't end with zip
# https://gitlab.com/user/reponame/repository/archive.zip?ref=b%C3%A9po


def resolve_known_host_service(url, name, branch, *, reporter=None):
    """Determine source type and full URL for known hosting service

    If the service is not determined from the provided URL, a tuple with
    two ``None`` values is returned.

    :param url: URL
    :param name: module name
    """
    if reporter is None:
        reporter = NullReporter()
    match = None
    actual_start = None
    for key, value in KNOWN_HOST_SERVICES_INFO.items():
        for start in value["possible_starts"]:
            if url.startswith(start + value["domain"]):
                match = value
                actual_start = start
                reporter.verbose(
                    _("Identified {0} as known hosting service").format(key)
                )
                for suffix in value["ignored_suffixes"]:
                    if url.endswith(suffix):
                        reporter.verbose(
                            _(
                                "Not using {service} as known hosting service"
                                " because the URL ends with '{suffix}'"
                            ).format(service=key, suffix=suffix)
                        )
                        return None, None
    if match:
        actual_start = match["url_start"] if not actual_start else ""
        if "branch" in match["url_end"]:
            suffix = match["url_end"].format(
                name=name,
                branch=branch or get_default_branch(url),
            )
        else:
            suffix = match["url_end"].format(name=name)
        url = "{prefix}{base}{suffix}".format(
            prefix=actual_start, base=url.rstrip("/"), suffix=suffix
        )
        reporter.verbose(_("Will use the following URL for download: {0}").format(url))
        return "remote_zip", url
    return None, None


def validate_url(url):
    if not Path(url).exists():
        url_validated = False
        message = None
        if url.startswith("http"):
            try:
                open_url = _urlopen(url)
                open_url.close()
                url_validated = True
            except URLError as error:
                message = error
        else:
            try:
                open_url = _urlopen("http://" + url)
                open_url.close()
                url_validated = True
            except URLError as error:
                message = error
            try:
                open_url = _urlopen("https://" + url)
                open_url.close()
                url_validated = True
            except URLError as error:
                message = error
        if not url_validated:
            raise SourceResolutionError(
                _("Cannot open URL <{url}>: {error}").format(url=url, error=message)
            )
    return True


# TODO: add also option to enforce the source type
# TODO: workaround, https://github.com/OSGeo/grass-addons/issues/528
def resolve_source_code(url=None, name=None, branch=None, fork=False, *, reporter=None):
    """Return type and URL or path of the source code

    Local paths are not presented as URLs to be usable in standard functions.
    Path is identified as local path if the directory of file exists which
    has the unfortunate consequence that the not existing files are evaluated
    as remote URLs. A remote URL which is not recognized as any specific
    source resolves to the "url" type: a plain URL which is usable as a
    metadata location for listing addons, but is not installable.
    When GitHub repository is specified, ZIP file link is returned. The ZIP
    is for {branch} branch, not the default one because GitHub does not
    provide the default branch in the URL (July 2015).

    :returns: tuple with type of source and full URL or path

    Official repository:

    >>> resolve_source_code(name="g.example")  # doctest: +SKIP
    ('official', 'https://trac.osgeo.org/.../general/g.example')

    Plain URLs:

    >>> resolve_source_code("https://example.com/addons/")  # doctest: +SKIP
    ('url', 'https://example.com/addons/')

    ZIP files online:

    >>> resolve_source_code(
    ...     "https://trac.osgeo.org/.../r.modis?format=zip"
    ... )  # doctest: +SKIP
    ('remote_zip', 'https://trac.osgeo.org/.../r.modis?format=zip')

    Local directories and ZIP files:

    >>> resolve_source_code(os.path.expanduser("~"))  # doctest: +ELLIPSIS
    ('dir', '...')
    >>> resolve_source_code("/local/directory/downloaded.zip")  # doctest: +SKIP
    ('zip', '/local/directory/downloaded.zip')

    OSGeo Trac:

    >>> resolve_source_code("trac.osgeo.org/.../r.agent.aco")  # doctest: +SKIP
    ('remote_zip', 'https://trac.osgeo.org/.../r.agent.aco?format=zip')
    >>> resolve_source_code("https://trac.osgeo.org/.../r.agent.aco")  # doctest: +SKIP
    ('remote_zip', 'https://trac.osgeo.org/.../r.agent.aco?format=zip')

    GitHub:

    >>> resolve_source_code("github.com/user/g.example")  # doctest: +SKIP
    ('remote_zip', 'https://github.com/user/g.example/archive/master.zip')
    >>> resolve_source_code("github.com/user/g.example/")  # doctest: +SKIP
    ('remote_zip', 'https://github.com/user/g.example/archive/master.zip')
    >>> resolve_source_code("https://github.com/user/g.example")  # doctest: +SKIP
    ('remote_zip', 'https://github.com/user/g.example/archive/master.zip')
    >>> resolve_source_code("https://github.com/user/g.example/")  # doctest: +SKIP
    ('remote_zip', 'https://github.com/user/g.example/archive/master.zip')

    GitLab:

    >>> resolve_source_code("gitlab.com/JoeUser/GrassModule")  # doctest: +SKIP
    ('remote_zip', 'https://gitlab.com/JoeUser/GrassModule/-/archive/master/GrassModule-master.zip')
    >>> resolve_source_code("https://gitlab.com/JoeUser/GrassModule")  # doctest: +SKIP
    ('remote_zip', 'https://gitlab.com/JoeUser/GrassModule/-/archive/master/GrassModule-master.zip')

    Bitbucket:

    >>> resolve_source_code("bitbucket.org/joe-user/grass-module")  # doctest: +SKIP
    ('remote_zip', 'https://bitbucket.org/joe-user/grass-module/get/default.zip')
    >>> resolve_source_code(
    ...     "https://bitbucket.org/joe-user/grass-module"
    ... )  # doctest: +SKIP
    ('remote_zip', 'https://bitbucket.org/joe-user/grass-module/get/default.zip')
    """  # noqa: E501
    # Handle URL for the official repo
    if not url or url == OFFICIAL_REPOSITORY_URL:
        return "official", OFFICIAL_REPOSITORY_URL

    # Check if URL can be found
    # Catch corner case if local URL is given starting with file://
    url = url.removeprefix("file://")
    validate_url(url)

    # Return validated URL for official fork
    if fork:
        return "official_fork", url

    # Handle local URLs
    if Path(url).is_dir():
        return "dir", os.path.abspath(url)
    if Path(url).exists():
        if url.endswith(".zip"):
            return "zip", os.path.abspath(url)
        for suffix in SUPPORTED_TAR_FORMATS:
            if url.endswith("." + suffix):
                return suffix, os.path.abspath(url)
        raise SourceResolutionError(
            _(
                "Local file <{path}> is not a supported archive."
                " Supported formats are: {formats}."
            ).format(path=url, formats=", ".join(("zip", *SUPPORTED_TAR_FORMATS)))
        )
    # Handle remote URLs
    source, resolved_url = resolve_known_host_service(
        url, name, branch, reporter=reporter
    )
    if source:
        return source, resolved_url
    # we allow URL to end with =zip or ?zip and not only .zip
    # unfortunately format=zip&version=89612 would require something else
    # special option to force the source type would solve it
    if url.endswith("zip"):
        return "remote_zip", url
    for suffix in SUPPORTED_TAR_FORMATS:
        if url.endswith(suffix):
            return "remote_" + suffix, url
    # A plain URL usable as a metadata location for listing addons,
    # but not installable.
    return "url", url
