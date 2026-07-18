#!/usr/bin/env python3

############################################################################
#
# MODULE:       g.extension
# AUTHOR(S):    Markus Neteler (original shell script)
#               Martin Landa <landa.martin gmail com> (Pythonized & upgraded for GRASS 7)
#               Vaclav Petras <wenzeslaus gmail com> (support for general sources)
# PURPOSE:      Tool to download and install extensions into local installation
#
# COPYRIGHT:    (C) 2009-2025 by Markus Neteler, and the GRASS Development Team
#
#               This program is free software under the GNU General
#               Public License (>=v2). Read the file COPYING that
#               comes with GRASS for details.
#
# TODO:         - update temporary workaround of using grass7 subdir of addon-repo, see
#                 https://github.com/OSGeo/grass-addons/issues/528
#               - add sudo support where needed (i.e. check first permission to write
#                 into $GISBASE directory)
#               - fix toolbox support in install_private_extension_xml()
#############################################################################

# %module
# % label: Maintains GRASS Addons extensions in local GRASS installation.
# % description: Downloads and installs extensions from GRASS Addons repository or other source into the local GRASS installation or removes installed extensions.
# % keyword: general
# % keyword: installation
# % keyword: extensions
# % keyword: addons
# % keyword: download
# %end

# %option
# % key: extension
# % type: string
# % key_desc: name
# % label: Name of extension to install or remove
# % description: Name of toolbox (set of extensions) when -t flag is given
# % required: yes
# %end
# %option
# % key: operation
# % type: string
# % description: Operation to be performed
# % required: yes
# % options: add,remove
# % answer: add
# %end
# %option
# % key: url
# % type: string
# % key_desc: url
# % label: URL or directory to get the extension from (supported only on Linux and Mac)
# % description: The official repository is used by default. User can specify a ZIP file, directory or a repository on common hosting services. See manual for all options.
# %end
# %option
# % key: prefix
# % type: string
# % key_desc: path
# % description: Prefix where to install extension (ignored when flag -s is given)
# % answer: $GRASS_ADDON_BASE
# % required: no
# %end
# %option
# % key: proxy
# % type: string
# % key_desc: proxy
# % description: Set the proxy with: "http=<value>,ftp=<value>"
# % required: no
# % multiple: yes
# %end
# %option
# % key: branch
# % type: string
# % key_desc: branch
# % description: Specific branch to fetch addon from (only used when fetching from git)
# % required: no
# % multiple: no
# %end

# %flag
# % key: l
# % description: List available extensions in the official GRASS Addons repository
# % guisection: Print
# % suppress_required: yes
# %end
# %flag
# % key: c
# % description: List available extensions in the official GRASS Addons repository including module description
# % guisection: Print
# % suppress_required: yes
# %end
# %flag
# % key: g
# % description: List available extensions in the official GRASS Addons repository (shell script style)
# % guisection: Print
# % suppress_required: yes
# %end
# %flag
# % key: a
# % description: List locally installed extensions
# % guisection: Print
# % suppress_required: yes
# %end
# %flag
# % key: s
# % description: Install system-wide (may need system administrator rights)
# % guisection: Install
# %end
# %flag
# % key: d
# % description: Download source code and exit
# % guisection: Install
# %end
# %flag
# % key: i
# % description: Do not install new extension, just compile it
# % guisection: Install
# %end
# %flag
# % key: f
# % description: Force removal when uninstalling extension (operation=remove)
# % guisection: Remove
# %end
# %flag
# % key: t
# % description: Operate on toolboxes instead of single modules (experimental)
# % suppress_required: yes
# %end
# %flag
# % key: o
# % description: url refers to a fork of the official extension repository
# %end

# %rules
# % required: extension, -l, -c, -g, -a
# % exclusive: extension, -l, -c, -g
# % exclusive: extension, -l, -c, -a
# % requires: -o, url
# % requires: branch, url
# %end

# TODO: solve addon-extension(-module) confusion

import fileinput
import os
import codecs
import sys
import re
import atexit
import shutil
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from subprocess import PIPE
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin

# Get the XML parsing exceptions to catch. The behavior changed with Python 2.7
# and ElementTree 1.3.
from xml.parsers import expat  # TODO: works for any Python?

if hasattr(ET, "ParseError"):
    ETREE_EXCEPTIONS = (ET.ParseError, expat.ExpatError)
else:
    ETREE_EXCEPTIONS = expat.ExpatError

import grass.script as gs
from grass.script.utils import try_rmdir
from grass.app.runtime import RuntimePaths
from grass.addons import config as addons_config
from grass.addons import resolve as addons_resolve
from grass.addons.config import (
    HTTP_HEADERS as HEADERS,
    OFFICIAL_REPOSITORY_URL as GIT_URL,
    default_make_program,
)
from grass.addons.exceptions import AddonsError
from grass.addons.registry import LocalRegistry, etree_fromfile, get_optional_params

# temp dir
REMOVE_TMPDIR = True
PROXIES = {}

MAKE = default_make_program()

copy_tree = partial(shutil.copytree, dirs_exist_ok=True)


class GrassReporter:
    """Reporter forwarding grass.addons messages to grass.script messaging"""

    def message(self, text):
        gs.message(text)

    def verbose(self, text):
        gs.verbose(text)

    def debug(self, text):
        gs.debug(text, 1)

    def warning(self, text):
        gs.warning(text)

    def check_cancelled(self):
        pass


REPORTER = GrassReporter()


def _registry():
    """Create access to the metadata files in the install prefix"""
    return LocalRegistry(
        options["prefix"],
        version_major=VERSION[0],
        libgis_revision=gs.version()["libgis_revision"],
        reporter=REPORTER,
    )


class GitAdapter:
    """
    Basic class for listing and downloading GRASS AddOns using git

    """

    def __init__(
        self,
        addons=[],
        url="https://github.com/osgeo/grass-addons",
        git="git",
        working_directory=None,
        official_repository_structure=True,
        major_grass_version=None,
        branch=None,
        verbose=False,
        quiet=False,
    ):
        #: Attribute containing list of addons names
        self._addons = addons
        #: Attribute containing Git command name
        self._git = git
        #: Attribute containing the URL to the online repository
        self.url = url
        self.major_grass_version = major_grass_version
        #: Attribute flagging if the repository is structured like the official addons
        # repository
        self.official_repository_structure = official_repository_structure
        #: Attribute containing the path to the working directory where the repo is
        # cloned out to
        if working_directory:
            self.working_directory = Path(working_directory).absolute()
        else:
            self.working_directory = Path().absolute()

        # Check if working directory is writable
        self.__check_permissions()
        # Check if Git is installed
        self._is_git_installed()

        #: Attribute containing available branches
        self.branches = self._get_branch_list()
        #: Attribute containing the git version
        self.git_version = self._get_version()
        # Initialize the local copy
        self._initialize_clone()
        #: Attribute containing the default branch of the repository
        self.default_branch = self._get_default_branch()
        #: Attribute containing the branch used for checkout
        self.branch = self._set_branch(branch)
        #: Attribute containing list of addons in the repository with path to
        # directories
        self.addons = self._get_addons_list()

    def _get_version(self):
        """Get the installed git version"""
        git_version = gs.Popen([self._git, "--version"], stdout=PIPE, stderr=PIPE)
        git_version, stderr = git_version.communicate()
        if stderr:
            gs.fatal(
                _("Failed to get Git version.\n{}").format(
                    gs.decode(stderr),
                )
            )
        git_version = re.search(r"\d+.(\d+.\d+|\d+)", gs.decode(git_version))
        if not git_version:
            gs.fatal(_("Failed to get Git version."))
        git_version = git_version.group()
        if git_version.count(".") == 2:
            git_version = git_version.rsplit(".", 1)[0]
        return float(git_version)

    def _initialize_clone(self):
        """Get a minimal working copy of a git repository without content"""
        repo_directory = "grass_addons"
        if not self.working_directory.exists():
            self.working_directory.mkdir(exist_ok=True, parents=True)
        gs.call(
            [
                self._git,
                "clone",
                "-q",
                "--no-checkout",
                "--filter=blob:none",
                self.url,
                repo_directory,
            ],
            cwd=self.working_directory,
        )
        self.local_copy = self.working_directory / repo_directory

    def _is_git_installed(self):
        """Check if Git command is installed"""
        try:
            gs.call([self._git], stdout=PIPE)
        except OSError:
            gs.fatal(_("Could not found Git. Please install it."))

    def __check_permissions(self):
        """"""
        # Create working directory if it does not exist
        self.working_directory.mkdir(parents=True, exist_ok=True)
        # Check pemissions in case he workdir existed
        if not os.access(self.working_directory, os.W_OK):
            gs.fatal(
                _("Cannot write to working directory {}.").format(
                    self.working_directory
                )
            )

    def _get_branch_list(self):
        """Return commit hash reference and names for remote branches of
        a git repository

        :param url: URL to git repository, defaults to the official GRASS
                    addon repository
        """
        branch_list = gs.Popen(
            [self._git, "ls-remote", "--heads", self.url],
            stdout=PIPE,
        )
        branch_list = gs.decode(branch_list.communicate()[0])
        return {
            branch.rsplit("/", 1)[-1]: branch.split("\t", 1)[0]
            for branch in branch_list.split("\n")
        }

    def _get_default_branch(self):
        """Return commit hash reference and names for remote branches of
        a git repository

        :param url: URL to git repository, defaults to the official GRASS
                    addon repository
        """
        default_branch = gs.Popen(
            [self._git, "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=self.local_copy,
            stdout=PIPE,
        )
        return gs.decode(default_branch.communicate()[0]).rstrip().rsplit("/", 1)[-1]

    def _get_version_branch(self):
        """Check if version branch for the current GRASS version exists,
        The method is only useful for repositories that follow the structure
        and concept of the official addon repository

        Returns None if no version branch is found."""
        version_branch = f"grass{self.major_grass_version}"

        return version_branch if version_branch in self.branches else None

    def _set_branch(self, branch_name):
        """Set the branch to check out to either:
        a) a user defined branch
        b) a version branch for repositories following the official addons repository
           structure
        c) the default branch of the repository
        """
        checkout_branch = None
        # Check user provided branch
        if branch_name:
            if branch_name not in self.branches:
                gs.fatal(
                    _("Branch <{branch}> not found in repository <{url}>").format(
                        branch=branch_name, url=self.url
                    )
                )
            else:
                checkout_branch = branch_name
        # Check version branch if relevant
        elif self.official_repository_structure:
            checkout_branch = self._get_version_branch()

        # Use default branch if none of the above are found
        return checkout_branch or self.default_branch

    def _get_addons_list(self):
        """Build a dictionary with addon name as key and path to directory with
        Makefile in repository"""
        file_list = gs.Popen(
            [self._git, "ls-tree", "--name-only", "-r", self.branch],
            cwd=self.local_copy,
            stdout=PIPE,
            stderr=PIPE,
        )
        file_list, stderr = file_list.communicate()
        if stderr:
            gs.fatal(
                _(
                    "Failed to get addons files list from the"
                    " Git repository <{repo_path}>.\n{error}"
                ).format(
                    repo_path=self.local_copy,
                    error=gs.decode(stderr),
                )
            )
        # Build addons dict
        addons_dict = {}
        addons = [f".*{addon}/Makefile\n" for addon in self._addons]
        addons_makefile_pattern = re.compile(
            rf"({'|'.join(addons)})",
        )
        addons_makefiles = re.findall(
            addons_makefile_pattern,
            gs.decode(file_list),
        )
        for addon in addons_makefiles:
            addon_dir = os.path.dirname(addon)
            addons_dict[os.path.basename(addon_dir)] = addon_dir.rstrip()
        return addons_dict

    def _addon_exists(self, addon_list):
        if not [self.addons[addon] for addon in addon_list if addon in self.addons]:
            gs.fatal(
                _("No addon named <{}> found in the repository.").format(
                    ", ".join(addon_list)
                )
            )

    def fetch_addons(self, addon_list, all_addons=False):
        self._addon_exists(addon_list)
        if addon_list:
            if self.git_version >= 2.25 and not all_addons:
                gs.call(
                    [self._git, "sparse-checkout", "init", "--cone"],
                    cwd=self.local_copy,
                )
                gs.call(
                    [
                        self._git,
                        "sparse-checkout",
                        "set",
                        *[self.addons[addon] for addon in addon_list],
                    ],
                    cwd=self.local_copy,
                )
        gs.call(
            [self._git, "checkout", self.branch],
            cwd=self.local_copy,
        )

    def get_addons_src_code_git_repo_url_path(self):
        """Get addons official GitHub repository source code URL path

        :return dict addons_url: dictionary of addons official GitHub
                                 repository source code URL path
        """
        addons_url = {}
        for addon in self.addons:
            addons_url[addon] = urljoin(
                self.url,
                urljoin(
                    "tree/",
                    urljoin(
                        f"{self.branch}/",
                        self.addons[addon],
                    ),
                ),
            )
        return addons_url


def replace_shebang_win(python_file):
    """
    Replaces "python" with "python3" in python files
    using UTF8 encoding on MS Windows
    """

    cur_dir = os.path.dirname(python_file)
    tmp_name = os.path.join(cur_dir, gs.tempname(12))

    with (
        codecs.open(python_file, "r", encoding="utf8") as in_file,
        codecs.open(tmp_name, "w", encoding="utf8") as out_file,
    ):
        for line in in_file:
            new_line = line.replace(
                "#!/usr/bin/env python\n", "#!/usr/bin/env python3\n"
            )
            out_file.write(new_line)

    os.remove(python_file)  # remove original
    Path(tmp_name).rename(python_file)  # rename temp to original name


def urlretrieve(url, filename, *args, **kwargs):
    """Same function as 'urlretrieve', but with the ability to
    define headers.
    """
    request = urlrequest.Request(url, headers=HEADERS)
    response = urlrequest.urlopen(request, *args, **kwargs)
    Path(filename).write_bytes(response.read())


def urlopen(url, *args, **kwargs):
    """Wrapper around urlopen. Same function as 'urlopen', but with the
    ability to define headers.
    """
    request = urlrequest.Request(url, headers=HEADERS)
    return urlrequest.urlopen(request, *args, **kwargs)


def get_version_branch(major_version):
    """Get the addons repository branch for the given GRASS major version"""
    try:
        return addons_resolve.get_version_branch(major_version)
    except AddonsError as error:
        gs.fatal(str(error))


def etree_fromurl(url):
    """Create XML element tree from a given URL"""
    try:
        file_ = urlopen(url)
    except URLError:
        gs.fatal(
            _(
                "Download file from <{url}>,"
                " failed. File is not on the server or"
                " check your internet connection."
            ).format(url=url),
        )
    return ET.fromstring(file_.read())


def check_progs():
    """Check if the necessary programs are available"""
    for prog in (MAKE, "gcc", "git"):
        if not gs.find_program(prog, "--help"):
            gs.fatal(_("'%s' required. Please install '%s' first.") % (prog, prog))


def get_installed_extensions(force=False):
    """Get list of installed extensions or toolboxes (if -t is set)"""
    if flags["t"]:
        return _registry().get_installed_toolboxes(force)

    # TODO: extension != module
    return _registry().get_installed_modules(force, shell_format=flags["g"])


def list_installed_extensions(toolboxes=False):
    """List installed extensions"""
    elist = get_installed_extensions()
    if elist:
        if toolboxes:
            gs.message(_("List of installed extensions (toolboxes):"))
        else:
            gs.message(_("List of installed extensions (modules):"))
        sys.stdout.write("\n".join(elist))
        sys.stdout.write("\n")
    elif toolboxes:
        gs.info(_("No extension (toolbox) installed"))
    else:
        gs.info(_("No extension (module) installed"))


# list extensions (read XML file from gs.osgeo.org/addons)


def list_available_extensions(url):
    """List available extensions/modules or toolboxes (if -t is given)

    For toolboxes it lists also all modules.
    """
    gs.debug("list_available_extensions(url={0})".format(url))
    if flags["t"]:
        gs.message(_("List of available extensions (toolboxes):"))
        tlist = get_available_toolboxes(url)
        tkeys = sorted(tlist.keys())
        for toolbox_code in tkeys:
            toolbox_data = tlist[toolbox_code]
            if flags["g"]:
                print("toolbox_name=" + toolbox_data["name"])
                print("toolbox_code=" + toolbox_code)
            else:
                print("%s (%s)" % (toolbox_data["name"], toolbox_code))
            if flags["c"] or flags["g"]:
                list_available_modules(url, toolbox_data["modules"])
            elif toolbox_data["modules"]:
                print(os.linesep.join(["* " + x for x in toolbox_data["modules"]]))
    else:
        gs.message(_("List of available extensions (modules):"))
        # TODO: extensions with several modules + lib
        list_available_modules(url)


def get_available_toolboxes(url):
    """Return toolboxes available in the repository"""
    tdict = {}
    url += "toolboxes.xml"
    try:
        tree = etree_fromurl(url)
        for tnode in tree.findall("toolbox"):
            mlist = []
            clist = []
            tdict[tnode.get("code")] = {
                "name": tnode.get("name"),
                "correlate": clist,
                "modules": mlist,
            }

            for cnode in tnode.findall("correlate"):
                clist.append(cnode.get("name"))

            for mnode in tnode.findall("task"):
                mlist.append(mnode.get("name"))
    except (HTTPError, OSError):
        gs.fatal(_("Unable to fetch addons metadata file"))

    return tdict


def get_toolbox_extensions(url, name):
    """Get extensions inside a toolbox in toolbox file at given URL

    :param url: URL of the directory (file name will be attached)
    :param name: toolbox name
    """
    # dictionary of extensions
    edict = {}

    url += "toolboxes.xml"

    try:
        tree = etree_fromurl(url)
        for tnode in tree.findall("toolbox"):
            if name == tnode.get("code"):
                for enode in tnode.findall("task"):
                    # extension name
                    ename = enode.get("name")
                    edict[ename] = {}
                    # list of modules installed by this extension
                    edict[ename]["mlist"] = []
                    # list of files installed by this extension
                    edict[ename]["flist"] = []
                break
    except (HTTPError, OSError):
        gs.fatal(_("Unable to fetch addons metadata file"))

    return edict


def list_available_modules(url, mlist=None):
    """List modules available in the repository

    Tries to use XML metadata file first. Fallbacks to HTML page with a list.

    :param url: URL of the directory (file name will be attached)
    :param mlist: list only modules in this list
    """
    file_url = url + "modules.xml"
    gs.debug("url=%s" % file_url, 1)
    try:
        tree = etree_fromurl(file_url)
    except ETREE_EXCEPTIONS:
        gs.fatal(_("Unable to parse '{url}'").format(url=file_url))
    except (HTTPError, URLError, OSError) as error:
        gs.fatal(_("Unable to read '{url}': {error}").format(url=file_url, error=error))

    for mnode in tree.findall("task"):
        name = mnode.get("name").strip()
        if mlist and name not in mlist:
            continue
        if flags["c"] or flags["g"]:
            desc, keyw = get_optional_params(mnode)

        if flags["g"]:
            print("name=" + name)
            print("description=" + desc)
            print("keywords=" + keyw)
        elif flags["c"]:
            if mlist:
                print("*", end="")
            print(name + " - " + desc)
        else:
            print(name)


def cleanup():
    """Cleanup after the downloads and compilation"""
    if REMOVE_TMPDIR:
        try_rmdir(TMPDIR)


def write_xml_modules(name, tree=None):
    """Write element tree as a modules metadata file"""
    _registry().write_xml_modules(name, tree)


def write_xml_extensions(name, tree=None):
    """Write element tree as an extensions metadata file"""
    _registry().write_xml_extensions(name, tree)


def install_extension(source=None, url=None, xmlurl=None, branch=None):
    """Install extension (e.g. one module) or a toolbox (list of modules)"""
    gisbase = os.getenv("GISBASE")
    if not gisbase:
        gs.fatal(_("$GISBASE not defined"))

    if options["extension"] in get_installed_extensions(force=True):
        gs.warning(
            _("Extension <%s> already installed. Re-installing...")
            % options["extension"]
        )

    # create a dictionary of extensions
    # for each extension
    #   - a list of modules installed by this extension
    #   - a list of files installed by this extension

    edict = None
    if flags["t"]:
        gs.message(_("Installing toolbox <%s>...") % options["extension"])
        edict = get_toolbox_extensions(xmlurl, options["extension"])
    else:
        edict = {}
        edict[options["extension"]] = {}
        # list of modules installed by this extension
        edict[options["extension"]]["mlist"] = []
        # list of files installed by this extension
        edict[options["extension"]]["flist"] = []
    if not edict:
        gs.warning(_("Nothing to install"))
        return

    ret = 0
    tmp_dir = None

    new_modules = []
    for extension in edict:
        ret1 = 0
        new_modules_ext = None
        if sys.platform == "win32":
            ret1, new_modules_ext, new_files_ext = install_extension_win(extension)
        else:
            (
                ret1,
                new_modules_ext,
                new_files_ext,
                tmp_dir,
            ) = install_extension_std_platforms(
                extension, source=source, url=url, branch=branch
            )
        if not flags["d"] and not flags["i"]:
            edict[extension]["mlist"].extend(new_modules_ext)
            edict[extension]["flist"].extend(new_files_ext)
            new_modules.extend(new_modules_ext)
            ret += ret1
            if len(edict) > 1:
                print("-" * 60)

    if flags["d"] or flags["i"]:
        return

    if ret != 0:
        gs.warning(_("Installation failed, sorry. Please check above error messages."))
    else:
        # update extensions metadata file
        gs.message(_("Updating extensions metadata file..."))
        install_extension_xml(edict)

        # update modules metadata file
        gs.message(_("Updating extension modules metadata file..."))
        install_module_xml(new_modules, source=source)

        for module in new_modules:
            update_manual_page(module, source=source)

        gs.message(
            _("Installation of <%s> successfully finished") % options["extension"]
        )

    if not os.getenv("GRASS_ADDON_BASE"):
        gs.warning(
            _(
                "This add-on module will not function until"
                " you set the GRASS_ADDON_BASE environment"
                ' variable (see "g.manual variables")'
            )
        )


def get_toolboxes_metadata(url):
    """Return metadata for all toolboxes from given URL

    :param url: URL of a modules metadata file
    :param mlist: list of modules to get metadata for
    :returns: tuple where first item is dictionary with module names as keys
        and dictionary with dest, keyw, files keys as value, the second item
        is list of 'binary' files (installation files)
    """
    data = {}
    try:
        tree = etree_fromurl(url)
        for tnode in tree.findall("toolbox"):
            clist = []
            for cnode in tnode.findall("correlate"):
                clist.append(cnode.get("code"))

            mlist = []
            for mnode in tnode.findall("task"):
                mlist.append(mnode.get("name"))

            code = tnode.get("code")
            data[code] = {
                "name": tnode.get("name"),
                "correlate": clist,
                "modules": mlist,
            }
    except (HTTPError, OSError):
        gs.error(_("Unable to read addons metadata file from the remote server"))
    return data


def get_addons_metadata(url, mlist):
    """Return metadata for list of modules from given URL

    :param url: URL of a modules metadata file
    :param mlist: list of modules to get metadata for
    :returns: tuple where first item is dictionary with module names as keys
        and dictionary with dest, keyw, files keys as value, the second item
        is list of 'binary' files (installation files)
    """

    # TODO: extensions with multiple modules
    data = {}
    bin_list = []
    try:
        tree = etree_fromurl(url)
    except (HTTPError, URLError, OSError) as error:
        gs.error(
            _("Unable to read addons metadata file from the remote server: {0}").format(
                error
            )
        )
        return data, bin_list
    except ETREE_EXCEPTIONS as error:
        gs.warning(_("Unable to parse '%s': {0}").format(error) % url)
        return data, bin_list
    for mnode in tree.findall("task"):
        name = mnode.get("name")
        if name not in mlist:
            continue
        file_list = []
        bnode = mnode.find("binary")
        windows = sys.platform == "win32"
        if bnode is not None:
            for fnode in bnode.findall("file"):
                path = fnode.text.split("/")
                if path[0] == "bin":
                    bin_list.append(path[-1])
                    if windows:
                        path[-1] += ".exe"
                elif path[0] == "scripts":
                    bin_list.append(path[-1])
                    if windows:
                        path[-1] += ".py"
                file_list.append(os.path.sep.join(path))
        desc, keyw = get_optional_params(mnode)
        data[name] = {
            "desc": desc,
            "keyw": keyw,
            "files": file_list,
        }
    return data, bin_list


def install_extension_xml(edict):
    """Update the extensions metadata file with an installed extension"""
    _registry().install_extension_xml(edict)


def get_multi_addon_addons_which_install_only_html_man_page():
    """Get multi-addon addons which install only manual html page

    :return list addons: list of multi-addon addons which install
                         only manual html page
    """
    all_addon_dirs = []
    addon_paths = re.findall(
        rf".*{options['extension']}*.",
        get_addons_paths(gg_addons_base_dir=options["prefix"]),
    )
    addon_dir_paths = {os.path.dirname(i) for i in addon_paths}
    for addon_dir in addon_dir_paths:
        addon_src_files = list(
            re.finditer(rf"{addon_dir}/(.*py)|(.*c)\n", "\n".join(addon_paths)),
        )
        if not addon_src_files:
            all_addon_dirs.append(os.path.basename(addon_dir))
        else:
            for addon_src_file in addon_src_files:
                addon_paths.pop(addon_paths.index(addon_src_file.group(0)))
    return all_addon_dirs


def filter_multi_addon_addons(mlist):
    """Filter out list of multi-addon addons which contains
    and installs only *.html manual page, without source/binary
    executable module and doesn't need to check metadata.

    e.g. the i.sentinel multi-addon consists of several full i.sentinel.*
    addons along with a i.sentinel.html overview file.


    :param list mlist: list of multi-addons (groups of addons
                       with respective addon overview HTML pages)

    :return list mlist: list of individual multi-addons without respective
                        addon overview HTML pages
    """
    # Filters out add-ons that only contain the *.html man page,
    # e.g. multi-addon i.sentinel (root directory) contains only
    # the *.html manual page for installation, it does not need
    # to check if metadata is available if there is no executable module.
    for addon in get_multi_addon_addons_which_install_only_html_man_page():
        if addon in mlist:
            mlist.pop(mlist.index(addon))
    return mlist


def install_module_xml(mlist, source=None):
    """Update the modules metadata file with installed modules"""
    # Identifying multi-addon addons queries the official repository over
    # the network, so skip it for other sources (e.g. a local directory).
    if (
        sys.platform != "win32"
        and source in {"official", "official_fork"}
        and len(mlist) > 1
    ):
        # mlist.copy() keeps the original list of add-ons
        mlist = filter_multi_addon_addons(mlist.copy())

    _registry().install_module_xml(mlist)

    return mlist


def install_extension_win(name):
    """Install extension on MS Windows"""
    gs.message(
        _("Downloading precompiled GRASS Addons <{}>...").format(options["extension"])
    )

    # build base URL
    base_url = (
        "http://wingrass.fsv.cvut.cz/"
        f"grass{VERSION[0]}{VERSION[1]}/addons/"
        f"grass-{VERSION[0]}.{VERSION[1]}.{VERSION[2]}"
    )

    # resolve ZIP URL
    source, url = resolve_source_code(url="{0}/{1}.zip".format(base_url, name))

    # to hide non-error messages from subprocesses
    outdev = open(os.devnull, "w") if gs.verbosity() <= 2 else sys.stdout

    # download Addons ZIP file
    os.chdir(TMPDIR)  # this is just to not leave something behind
    srcdir = os.path.join(TMPDIR, name)
    download_source_code(
        source=source,
        url=url,
        name=name,
        outdev=outdev,
        directory=srcdir,
        tmpdir=TMPDIR,
    )

    # collect module names and file names
    module_list = []
    module_name_pattern = re.compile(
        r"^([d,g,i,m,p,r,s,t,v]|^db|^ps|^r3|^wx)\..*[\.py,\.exe]$"
    )
    for r, d, f in os.walk(srcdir):
        for file in f:
            # Filter GRASS module name patterns
            if re.search(module_name_pattern, file):
                modulename = os.path.splitext(file)[0]
                module_list.append(modulename)
    # remove duplicates in case there are .exe wrappers for python scripts
    module_list = set(module_list)

    # change shebang from python to python3
    pyfiles = []
    for r, d, f in os.walk(srcdir):
        for file in f:
            if file.endswith(".py"):
                pyfiles.append(os.path.join(r, file))

    for filename in pyfiles:
        replace_shebang_win(filename)

    # collect old files
    old_file_list = []
    for r, d, f in os.walk(options["prefix"]):
        for filename in f:
            fullname = os.path.join(r, filename)
            old_file_list.append(fullname)

    # copy Addons copy tree to destination directory
    move_extracted_files(
        extract_dir=srcdir, target_dir=options["prefix"], files=os.listdir(srcdir)
    )

    # collect new files
    file_list = []
    for r, d, f in os.walk(options["prefix"]):
        for filename in f:
            fullname = os.path.join(r, filename)
            if fullname not in old_file_list:
                file_list.append(fullname)

    return 0, module_list, file_list


def download_source_code_official_github(url, name, branch, directory=None):
    """Download source code from a official GitHub repository

    .. note::
        Stdout is passed to to *outdev* while stderr is just printed.

    :param url: URL of the repository
        (module class/family and name are attached)
    :param name: module name
    :param branch: branch of the git repository to fetch from
    :param directory: directory where the source code will be downloaded
        (default is the current directory with name attached)

    :return str, str: full path to the directory with the source code
                      (useful when you not specify directory, if
                      *directory* is specified the return value is equal
                      to it),
                      addon official GitHub repository source code URL path
    """

    try:
        ga = GitAdapter(
            addons=[name],
            url=url,
            working_directory=directory,
            major_grass_version=int(VERSION[0]),
            branch=branch,
        )
    except RuntimeError:
        gs.fatal(_("GRASS Addons <%s> not found") % name)

    ga.fetch_addons([name])

    return (
        str(ga.local_copy / ga.addons[name]),
        ga.get_addons_src_code_git_repo_url_path()[name],
    )


def move_extracted_files(extract_dir, target_dir, files):
    """Fix state of extracted files by moving them to different directory

    When extracting, it is not clear what will be the root directory
    or if there will be one at all. So this function moves the files to
    a different directory in the way that if there was one directory extracted,
    the contained files are moved.
    """
    gs.debug("move_extracted_files({0})".format(locals()))
    if len(files) == 1:
        shutil.copytree(os.path.join(extract_dir, files[0]), target_dir)
    else:
        Path(target_dir).mkdir(exist_ok=True)
        for file_name in files:
            actual_file = os.path.join(extract_dir, file_name)
            if Path(actual_file).is_dir():
                # shutil.copytree() replaced by copy_tree() because
                # shutil's copytree() fails when subdirectory exists
                copy_tree(actual_file, os.path.join(target_dir, file_name))
            else:
                shutil.copy(actual_file, os.path.join(target_dir, file_name))


# Original copyright and license of the original version of the CRLF function
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010
# Python Software Foundation; All Rights Reserved
# Python Software Foundation License Version 2
# http://svn.python.org/projects/python/trunk/Tools/scripts/crlf.py
def fix_newlines(directory):
    """Replace CRLF with LF in all files in the directory

    Binary files are ignored. Recurses into subdirectories.
    """
    # skip binary files
    # see https://stackoverflow.com/a/7392391
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})

    def is_binary_string(bytes):
        return bool(bytes.translate(None, textchars))

    for root, unused, files in os.walk(directory):
        for name in files:
            filename = os.path.join(root, name)
            if is_binary_string(open(filename, "rb").read(1024)):
                continue  # ignore binary files

            # read content of text file
            data = Path(filename).read_bytes()

            # we don't expect there would be CRLF file by
            # purpose if we want to allow CRLF files we would
            # have to whitelite .py etc
            newdata = data.replace(b"\r\n", b"\n")
            if newdata != data:
                Path(filename).write_bytes(newdata)


def extract_zip(name, directory, tmpdir):
    """Extract a ZIP file into a directory"""
    gs.debug(
        "extract_zip(name={name}, directory={directory}, tmpdir={tmpdir})".format(
            name=name, directory=directory, tmpdir=tmpdir
        ),
        3,
    )
    try:
        zip_file = zipfile.ZipFile(name, mode="r")
        file_list = zip_file.namelist()
        # we suppose we can write to parent of the given dir
        # (supposing a tmp dir)
        extract_dir = os.path.join(tmpdir, "extract_dir")
        Path(extract_dir).mkdir()
        for subfile in file_list:
            if "__pycache__" in subfile:
                continue
            zip_file.extract(subfile, extract_dir)
        files = os.listdir(extract_dir)
        move_extracted_files(extract_dir=extract_dir, target_dir=directory, files=files)
    except zipfile.BadZipfile as error:
        gs.fatal(_("ZIP file is unreadable: {0}").format(error))


# TODO: solve the other related formats
def extract_tar(name, directory, tmpdir):
    """Extract a TAR or a similar file into a directory"""
    gs.debug(
        "extract_tar(name={name}, directory={directory}, tmpdir={tmpdir})".format(
            name=name, directory=directory, tmpdir=tmpdir
        ),
        3,
    )
    import tarfile

    try:
        tar = tarfile.open(name)
        extract_dir = os.path.join(tmpdir, "extract_dir")
        Path(extract_dir).mkdir()

        # The 'data' extraction filter was added in Python 3.12 and backported
        # to 3.11.4 (PEP 706). Refuse to extract without it rather
        # than extracting unsafely.
        if not hasattr(tarfile, "data_filter"):
            gs.fatal(_("Extracting may be unsafe; upgrade Python to 3.11.4 or newer"))
        tar.extractall(path=extract_dir, filter="data")

        files = os.listdir(extract_dir)
        move_extracted_files(extract_dir=extract_dir, target_dir=directory, files=files)
    except tarfile.TarError as error:
        gs.fatal(_("Archive file is unreadable: {0}").format(error))

    del tarfile  # we don't need it anywhere else


extract_tar.supported_formats = ["tar.gz", "gz", "bz2", "tar", "gzip", "targz"]


def download_source_code(
    source, url, name, outdev, directory=None, tmpdir=None, branch=None
):
    """Get source code to a local directory for compilation

    :return dictionary, url: addon source code directory path,
                             addon official GitHub repository source code
                             URL path
    """
    gs.verbose(_("Type of source identified as '{source}'.").format(source=source))
    if source in {"official", "official_fork"}:
        gs.message(
            _("Fetching <{name}> from <{url}> (be patient)...").format(
                name=name, url=url
            )
        )
        directory, url = download_source_code_official_github(
            url, name, branch, directory=directory
        )
    elif source == "url":
        gs.fatal(
            _(
                "Installing from the plain URL <{url}> is not supported."
                " Provide a ZIP or tar archive URL, a URL of a repository"
                " on a known hosting service (GitHub, GitLab, Bitbucket,"
                " OSGeo Trac), or a local path."
            ).format(url=url)
        )
    elif source == "remote_zip":
        gs.message(
            _("Fetching <{name}> from <{url}> (be patient)...").format(
                name=name, url=url
            )
        )
        # we expect that the module.zip file is not by chance in the archive
        zip_name = os.path.join(tmpdir, "extension.zip")
        try:
            response = urlopen(url)
        except URLError:
            # Try download add-on from 'master' branch if default "main" fails
            if not branch:
                try:
                    url = url.replace("main", "master")
                    gs.message(
                        _(
                            "Expected default branch not found. "
                            "Trying again from <{url}>..."
                        ).format(url=url)
                    )
                    response = urlopen(url)
                except URLError:
                    gs.fatal(
                        _(
                            "Extension <{name}> not found. Please check "
                            "'url' and 'branch' options"
                        ).format(name=name)
                    )
            else:
                gs.fatal(_("Extension <{}> not found").format(name))

        with open(zip_name, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        extract_zip(name=zip_name, directory=directory, tmpdir=tmpdir)
        fix_newlines(directory)
    elif (
        source.startswith("remote_")
        and source.split("_")[1] in extract_tar.supported_formats
    ):
        # we expect that the module.tar.gz file is not by chance in the archive
        archive_name = os.path.join(tmpdir, "extension." + source.split("_")[1])
        urlretrieve(url, archive_name)
        extract_tar(name=archive_name, directory=directory, tmpdir=tmpdir)
        fix_newlines(directory)
    elif source == "zip":
        extract_zip(name=url, directory=directory, tmpdir=tmpdir)
        fix_newlines(directory)
    elif source in extract_tar.supported_formats:
        extract_tar(name=url, directory=directory, tmpdir=tmpdir)
        fix_newlines(directory)
    elif source == "dir":
        shutil.copytree(url, directory)
        fix_newlines(directory)
    else:
        # probably programmer error
        gs.fatal(
            _(
                "Unknown extension (addon) source type '{0}'."
                " Please report this to the grass-user mailing list."
            ).format(source)
        )
    assert Path(directory).is_dir()
    return directory, url


def create_md_if_missing(root_dir):
    """Recursively searches for HTML files in the specified directory.
    If an HTML file does not have a corresponding Markdown (.md) file,
    it creates one by copying the HTML file and renaming it.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        html_files = [f for f in filenames if f.endswith(".html")]

        for html_file in html_files:
            md_file = os.path.splitext(html_file)[0] + ".md"
            md_path = os.path.join(dirpath, md_file)

            if not Path(md_path).exists():
                html_path = os.path.join(dirpath, html_file)
                shutil.copy(html_path, md_path)


def install_extension_std_platforms(name, source, url, branch):
    """Install extension on standard platforms"""
    runtime_paths = RuntimePaths()
    gisbase = runtime_paths.gisbase
    path_to_src_code_message = _("Path to the source code:")

    is_cmake = runtime_paths.is_cmake_build

    # to hide non-error messages from subprocesses
    outdev = open(os.devnull, "w") if gs.verbosity() <= 2 else sys.stdout

    os.chdir(TMPDIR)  # this is just to not leave something behind
    srcdir = os.path.join(TMPDIR, name)
    srcdir, url = download_source_code(
        source,
        url,
        name,
        outdev,
        directory=srcdir,
        tmpdir=TMPDIR,
        branch=branch,
    )
    create_md_if_missing(srcdir)
    os.chdir(srcdir)

    pgm_not_found_message = _(
        "Module name not found. Check module Makefile syntax (PGM variable)."
    )
    # collect module names
    module_list = []

    if is_cmake:
        for r, d, f in os.walk(srcdir):
            for filename in f:
                if filename == "CMakeLists.txt":
                    # get the module name: project(<module name>)
                    with open(os.path.join(r, "CMakeLists.txt")) as fp:
                        for line in fp:
                            m = re.match(r"project\(\s*(\S+).*\)", line)
                            if m:
                                try:
                                    modulename = m.group(1)
                                    if modulename:
                                        if modulename not in module_list:
                                            module_list.append(modulename)
                                    else:
                                        gs.fatal(pgm_not_found_message)
                                except IndexError:
                                    gs.fatal(pgm_not_found_message)
    else:
        for r, d, f in os.walk(srcdir):
            for filename in f:
                if filename == "Makefile":
                    # get the module name: PGM = <module name>
                    with open(os.path.join(r, "Makefile")) as fp:
                        for line in fp:
                            if re.match(r"PGM.*.=|PGM=", line):
                                try:
                                    modulename = line.split("=")[1].strip()
                                    if modulename:
                                        if modulename not in module_list:
                                            module_list.append(modulename)
                                    else:
                                        gs.fatal(pgm_not_found_message)
                                except IndexError:
                                    gs.fatal(pgm_not_found_message)

    # change shebang from python to python3
    pyfiles = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(srcdir):
        for file in f:
            if file.endswith(".py"):
                pyfiles.append(os.path.join(r, file))

    for filename in pyfiles:
        with fileinput.FileInput(filename, inplace=True) as file:
            for line in file:
                print(
                    line.replace("#!/usr/bin/env python\n", "#!/usr/bin/env python3\n"),
                    end="",
                )

    if is_cmake:
        grass_addon_base = options["prefix"]
        cmake_prefix_path = (
            ";" + os.getenv("CMAKE_PREFIX_PATH")
            if os.getenv("CMAKE_PREFIX_PATH")
            else ""
        )
        cmake_module_path = (
            ";" + os.getenv("CMAKE_MODULE_PATH")
            if os.getenv("CMAKE_MODULE_PATH")
            else ""
        )
        grass_cmake_prefix_path = (
            ";" + runtime_paths.grass_cmake_prefix_path
            if runtime_paths.grass_cmake_prefix_path
            else ""
        )
        g_cmake_config_dir = runtime_paths.grass_cmake_config_dir
        g_cmake_module_dir = runtime_paths.grass_cmake_module_dir
        g_c_compiler = os.getenv("CC") or runtime_paths.grass_cmake_c_compiler
        g_cxx_compiler = os.getenv("CXX") or runtime_paths.grass_cmake_cxx_compiler

        c_prefix_path = (
            f"{g_cmake_config_dir}{grass_cmake_prefix_path}{cmake_prefix_path}"
        )
        c_mod_path = f"{g_cmake_module_dir}{cmake_module_path}"
        c_compiler = f"-DCMAKE_C_COMPILER={g_c_compiler}" if g_c_compiler else ""
        cxx_compiler = (
            f"-DCMAKE_CXX_COMPILER={g_cxx_compiler}" if g_cxx_compiler else ""
        )

        config_cmd = [
            "cmake",
            "-B",
            "build",
            f"-DCMAKE_PREFIX_PATH={c_prefix_path}",
            f"-DCMAKE_MODULE_PATH={c_mod_path}",
            f"-DCMAKE_INSTALL_PREFIX={grass_addon_base}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DSOURCE_URL={url}",
            c_compiler,
            cxx_compiler,
        ]
        make_cmd = [
            "cmake",
            "--build",
            "build",
            "-v",
        ]
        install_cmd = [
            "cmake",
            "--install",
            "build",
        ]
        try:
            shutil.rmtree(os.path.join(srcdir, "build"))
        except FileNotFoundError:
            pass
    else:
        dirs = {
            "bin": os.path.join(srcdir, "bin"),
            "docs": os.path.join(srcdir, "docs"),
            "html": os.path.join(srcdir, "docs", "html"),
            "mkdocs": os.path.join(srcdir, "docs", "mkdocs"),
            "rest": os.path.join(srcdir, "docs", "rest"),
            "man": os.path.join(srcdir, "docs", "man"),
            "script": os.path.join(srcdir, "scripts"),
            # TODO: handle locales also for addons
            #             'string'  : os.path.join(srcdir, 'locale'),
            "string": srcdir,
            "etc": os.path.join(srcdir, "etc"),
        }
        make_cmd = [
            MAKE,
            "MODULE_TOPDIR=%s" % gisbase.replace(" ", r"\ "),
            "RUN_GISRC=%s" % os.environ["GISRC"],
            "BIN=%s" % dirs["bin"],
            "HTMLDIR=%s" % dirs["html"],
            "MDDIR=%s" % dirs["mkdocs"],
            "RESTDIR=%s" % dirs["rest"],
            "MANBASEDIR=%s" % dirs["man"],
            "SCRIPTDIR=%s" % dirs["script"],
            "STRINGDIR=%s" % dirs["string"],
            "ETC=%s" % os.path.join(dirs["etc"]),
            "SOURCE_URL=%s" % url,
        ]
        install_cmd = [
            MAKE,
            "MODULE_TOPDIR=%s" % gisbase,
            "ARCH_DISTDIR=%s" % srcdir,
            "INST_DIR=%s" % options["prefix"],
            "install",
        ]

    if flags["d"]:
        gs.message("\n%s\n" % _("To compile run:"))
        sys.stderr.write(" ".join(make_cmd) + "\n")
        gs.message("\n%s\n" % _("To install run:"))
        sys.stderr.write(" ".join(install_cmd) + "\n")
        gs.message(f"\n{path_to_src_code_message}\n")
        sys.stderr.write(f"{srcdir}\n")
        return 0, None, None, None

    os.chdir(srcdir)

    gs.message(_("Compiling..."))
    if not is_cmake and not Path(gisbase, "include", "Make", "Module.make").exists():
        gs.fatal(_("Please install GRASS development package"))

    if is_cmake and gs.call(config_cmd, stdout=outdev) != 0:
        gs.fatal(_("Compilation failed, sorry. Please check above error messages."))

    if gs.call(make_cmd, stdout=outdev) != 0:
        gs.fatal(_("Compilation failed, sorry. Please check above error messages."))

    if flags["i"]:
        gs.message(f"\n{path_to_src_code_message}\n")
        sys.stderr.write(f"{srcdir}\n")
        return 0, None, None, None

    # collect old files
    old_file_list = []
    for r, d, f in os.walk(options["prefix"]):
        for filename in f:
            fullname = os.path.join(r, filename)
            old_file_list.append(fullname)

    gs.message(_("Installing..."))
    ret = gs.call(install_cmd, stdout=outdev)

    # collect new files
    file_list = []
    for r, d, f in os.walk(options["prefix"]):
        for filename in f:
            fullname = os.path.join(r, filename)
            if fullname not in old_file_list:
                file_list.append(fullname)

    return ret, module_list, file_list, os.path.join(srcdir)


def remove_extension(force=False):
    """Remove existing extension
    extension or toolbox with extensions if -t is given)"""
    if flags["t"]:
        edict = get_toolbox_extensions(options["prefix"], options["extension"])
    else:
        edict = {}
        edict[options["extension"]] = {}
        # list of modules installed by this extension
        edict[options["extension"]]["mlist"] = []
        # list of files installed by this extension
        edict[options["extension"]]["flist"] = []

    # collect modules and files installed by these extensions
    mlist = []
    xml_file = os.path.join(options["prefix"], "extensions.xml")
    if Path(xml_file).exists():
        # read XML file
        tree = None
        try:
            tree = etree_fromfile(xml_file)
        except ETREE_EXCEPTIONS + (OSError, IOError):
            os.remove(xml_file)
            write_xml_extensions(xml_file)

        if tree is not None:
            for tnode in tree.findall("task"):
                ename = tnode.get("name").strip()
                if ename in edict:
                    # modules installed by this extension
                    mnode = tnode.find("modules")
                    if mnode is not None:
                        for fnode in mnode.findall("module"):
                            mname = fnode.text.strip()
                            edict[ename]["mlist"].append(mname)
                            mlist.append(mname)
                    # files installed by this extension
                    bnode = tnode.find("binary")
                    if bnode is not None:
                        for fnode in bnode.findall("file"):
                            bname = fnode.text.strip()
                            edict[ename]["flist"].append(bname)
    else:
        if force:
            write_xml_extensions(xml_file)

        xml_file = os.path.join(options["prefix"], "modules.xml")
        if not Path(xml_file).exists():
            if force:
                write_xml_modules(xml_file)
            else:
                gs.debug("No addons metadata file available", 1)

        # read XML file
        tree = None
        try:
            tree = etree_fromfile(xml_file)
        except ETREE_EXCEPTIONS + (OSError, IOError):
            os.remove(xml_file)
            write_xml_modules(xml_file)
            return []

        if tree is not None:
            for tnode in tree.findall("task"):
                ename = tnode.get("name").strip()
                if ename in edict:
                    # assume extension name == module name
                    edict[ename]["mlist"].append(ename)
                    mlist.append(ename)
                    # files installed by this extension
                    bnode = tnode.find("binary")
                    if bnode is not None:
                        for fnode in bnode.findall("file"):
                            bname = fnode.text.strip()
                            edict[ename]["flist"].append(bname)

    if force:
        gs.verbose(_("List of removed files:"))
    else:
        gs.info(_("Files to be removed:"))

    eremoved = remove_extension_files(edict, force)

    if force:
        if len(eremoved) > 0:
            gs.message(_("Updating addons metadata file..."))
            remove_extension_xml(mlist, edict)
            for ename in edict:
                if ename in eremoved:
                    gs.message(_("Extension <%s> successfully uninstalled.") % ename)
    elif flags["t"]:
        gs.warning(
            _("Toolbox <%s> not removed. Re-run '%s' with '-f' flag to force removal")
            % (options["extension"], "g.extension")
        )
    else:
        gs.warning(
            _("Extension <%s> not removed. Re-run '%s' with '-f' flag to force removal")
            % (options["extension"], "g.extension")
        )


# remove existing extension(s) (reading XML file)


def remove_extension_files(edict, force=False):
    """Remove extensions specified in a dictionary

    Uses the file names from the file list of the dictionary
    Fallbacks to standard layout of files on prefix path on error.
    """
    # try to read XML metadata file first
    xml_file = os.path.join(options["prefix"], "extensions.xml")

    einstalled = []
    eremoved = []

    if Path(xml_file).exists():
        tree = etree_fromfile(xml_file)
        if tree is not None:
            for task in tree.findall("task"):
                ename = task.get("name").strip()
                einstalled.append(ename)
    else:
        tree = None

    for name in edict:
        removed = True
        if len(edict[name]["flist"]) > 0:
            err = []
            for fpath in edict[name]["flist"]:
                gs.verbose(fpath)
                if force:
                    try:
                        os.remove(fpath)
                    except OSError:
                        msg = "Unable to remove file '%s'"
                        err.append(_(msg) % fpath)
                        removed = False
            if len(err) > 0:
                for error_line in err:
                    gs.error(error_line)
        else:
            if name not in einstalled:
                # try even if module does not seem to be available,
                # as the user may be trying to get rid of left over cruft
                gs.warning(_("Extension <%s> not found") % name)

            remove_extension_std(name, force)
            removed = False

        if removed is True:
            eremoved.append(name)

    return eremoved


def remove_extension_std(name, force=False):
    """Remove extension/module expecting the standard layout

    Any images for manuals or files installed in etc will not be
    removed
    """
    for fpath in [
        os.path.join(options["prefix"], "bin", name),
        os.path.join(options["prefix"], "scripts", name),
        os.path.join(options["prefix"], "docs", "html", name + ".html"),
        os.path.join(options["prefix"], "docs", "mkdocs", "source", name + ".md"),
        os.path.join(options["prefix"], "docs", "rest", name + ".txt"),
        os.path.join(options["prefix"], "docs", "man", "man1", name + ".1"),
    ]:
        if Path(fpath).is_file():
            gs.verbose(fpath)
            if force:
                os.remove(fpath)

    # remove module libraries under GRASS_ADDONS/etc/{name}/*
    libpath = os.path.join(options["prefix"], "etc", name)
    if Path(libpath).is_dir():
        gs.verbose(libpath)
        if force:
            shutil.rmtree(libpath)


def remove_extension_xml(mlist, edict):
    """Update local meta-file when removing existing extension"""
    _registry().remove_extension_xml(mlist, edict, options["extension"])


# check links in CSS


def check_style_file(name):
    """Ensures that a specified HTML documentation support file exists

    If the file, e.g. a CSS file does not exist, the file is copied from
    the distribution.

    If the files are missing, a warning is issued.
    """
    dist_file = os.path.join(os.getenv("GISBASE"), "docs", "html", name)
    addons_file = os.path.join(options["prefix"], "docs", "html", name)

    try:
        shutil.copyfile(dist_file, addons_file)
    except shutil.SameFileError:
        pass
    except OSError as error:
        gs.warning(
            _(
                "Unable to create '{filename}': {error}."
                " Is the GRASS documentation package installed?"
                " Installation continues,"
                " but documentation may not look right."
            ).format(filename=addons_file, error=error)
        )


def create_dir(path):
    """Creates the specified directory (with all dirs in between)

    NOOP for existing directory.
    """
    if Path(path).is_dir():
        return

    try:
        Path(path).mkdir(parents=True)
    except OSError as error:
        gs.fatal(_("Unable to create '%s': %s") % (path, error))

    gs.debug("'%s' created" % path)


def check_dirs():
    """Ensure that the necessary directories in prefix path exist"""
    create_dir(os.path.join(options["prefix"], "bin"))
    create_dir(os.path.join(options["prefix"], "docs", "html"))
    create_dir(os.path.join(options["prefix"], "docs", "mkdocs", "source"))
    create_dir(os.path.join(options["prefix"], "docs", "rest"))
    check_style_file("grass_logo.png")
    check_style_file("hamburger_menu.svg")
    check_style_file("hamburger_menu_close.svg")
    check_style_file("grassdocs.css")
    create_dir(os.path.join(options["prefix"], "etc"))
    create_dir(os.path.join(options["prefix"], "docs", "man", "man1"))
    create_dir(os.path.join(options["prefix"], "scripts"))


# fix file URI in manual page


def update_manual_page(module, source=None):
    """Fix manual page for addons which are at different directory
    than core modules"""
    if module.split(".", 1)[0] == "wx":
        return  # skip for GUI modules

    gs.verbose(_("Manual page for <%s> updated") % module)
    # read original html file
    htmlfile = os.path.join(options["prefix"], "docs", "html", module + ".html")
    try:
        oldfile = open(htmlfile)
        shtml = oldfile.read()
    except OSError as error:
        gs.fatal(_("Unable to read manual page: %s") % error)
    else:
        oldfile.close()

    pos = []

    # fix logo URL
    pattern = r'''<a href="([^"]+)"><img src="grass_logo.png"'''
    for match in re.finditer(pattern, shtml):
        if match.group(1)[:4] == "http":
            continue
        pos.append(match.start(1))

    # find URIs
    pattern = r"""<a href="([^"]+)">([^>]+)</a>"""
    addons = get_installed_extensions(force=True)
    # Identifying multi-addon addons queries the official repository over
    # the network, so skip it for other sources (e.g. a local directory).
    if sys.platform != "win32" and source in {"official", "official_fork"}:
        # Multi-addon
        if len(addons) > 1:
            for a in get_multi_addon_addons_which_install_only_html_man_page():
                # Add multi-addon addons which install only manual html page
                addons.append(a)

    for match in re.finditer(pattern, shtml):
        if match.group(1)[:4] == "http":
            continue
        if match.group(1).replace(".html", "") in addons:
            continue
        pos.append(match.start(1))

    if not pos:
        return  # no match

    # replace file URIs
    prefix = "file://" + "/".join([os.getenv("GISBASE"), "docs", "html"])
    ohtml = shtml[: pos[0]]
    for i in range(1, len(pos)):
        ohtml += prefix + "/" + shtml[pos[i - 1] : pos[i]]
    ohtml += prefix + "/" + shtml[pos[-1] :]

    # write updated html file
    try:
        newfile = open(htmlfile, "w")
        newfile.write(ohtml)
    except OSError as error:
        gs.fatal(_("Unable for write manual page: %s") % error)
    else:
        newfile.close()


def resolve_install_prefix(path, to_system):
    """Determine and check the path for installation"""
    try:
        result = addons_config.resolve_install_prefix(
            path,
            to_system,
            major_version=VERSION[0],
            minor_version=VERSION[1],
            env=os.environ,
            reporter=REPORTER,
        )
    except AddonsError as error:
        gs.fatal(str(error))
    os.environ["GRASS_PREFIX_ADDON_BASE"] = result  # make likes absolute paths
    return result


def resolve_xmlurl_prefix(url, source=None):
    """Determine and check the URL where the XML metadata files are stored"""
    try:
        return addons_resolve.resolve_xmlurl_prefix(
            url, source, major_version=VERSION[0], reporter=REPORTER
        )
    except AddonsError as error:
        gs.fatal(str(error))


def resolve_source_code(url=None, name=None, branch=None, fork=False):
    """Return type and URL or path of the source code"""
    try:
        return addons_resolve.resolve_source_code(
            url=url, name=name, branch=branch, fork=fork, reporter=REPORTER
        )
    except AddonsError as error:
        gs.fatal(str(error))


def get_addons_paths(gg_addons_base_dir):
    """Make or update list of the official addons source code paths
    prefix parameter plus /grass-addons directory using Git repository

    :param str gg_addons_base_dir: dir path where addons are installed

    :return str: list of all addons source code paths
    """
    addons_branch = get_version_branch(VERSION[0])
    grass_addons_dir = Path(gg_addons_base_dir) / "grass-addons"
    if grass_addons_dir.exists():
        try_rmdir(grass_addons_dir)
    gs.call(
        [
            "git",
            "clone",
            "-q",
            "--no-checkout",
            f"--branch={addons_branch}",
            "--filter=blob:none",
            GIT_URL,
        ],
        cwd=gg_addons_base_dir,
    )
    addons_file_list = gs.Popen(
        ["git", "ls-tree", "--name-only", "-r", addons_branch],
        cwd=grass_addons_dir,
        stdout=PIPE,
        stderr=PIPE,
    )
    addons_file_list, stderr = addons_file_list.communicate()
    if stderr:
        gs.fatal(
            _(
                "Failed to get addons files list from the"
                " Git repository <{repo_path}>.\n{error}"
            ).format(
                repo_path=grass_addons_dir,
                error=gs.decode(stderr),
            )
        )
    return gs.decode(addons_file_list)


def main():
    # check dependencies
    if not flags["a"] and sys.platform != "win32":
        check_progs()

    original_url = options["url"]
    branch = options["branch"]

    # manage proxies
    global PROXIES
    if options["proxy"]:
        PROXIES = {}
        for ptype, purl in (p.split("=") for p in options["proxy"].split(",")):
            PROXIES[ptype] = purl
        proxy = urlrequest.ProxyHandler(PROXIES)
        opener = urlrequest.build_opener(proxy)
        urlrequest.install_opener(opener)
        # Required for mkhtml.py script (get addon git commit from GitHub API server)
        os.environ["GRASS_PROXY"] = options["proxy"]

    # define path
    options["prefix"] = resolve_install_prefix(
        path=options["prefix"], to_system=flags["s"]
    )

    # list available extensions
    if flags["l"] or flags["c"] or (flags["g"] and not flags["a"]):
        # using dummy extension, we don't need any extension URL now,
        # but will work only as long as the function does not check
        # if the URL is actually valid or something
        source, url = resolve_source_code(
            name="dummy", url=original_url, branch=branch, fork=flags["o"]
        )
        xmlurl = resolve_xmlurl_prefix(original_url, source=source)
        list_available_extensions(xmlurl)
        return 0
    if flags["a"]:
        list_installed_extensions(toolboxes=flags["t"])
        return 0

    if flags["d"] or flags["i"]:
        flag = "d" if flags["d"] else "i"
        if options["operation"] != "add":
            gs.warning(
                _(
                    "Flag '{}' is relevant only to 'operation=add'. Ignoring this flag."
                ).format(flag)
            )
        else:
            global REMOVE_TMPDIR
            REMOVE_TMPDIR = False

    if options["operation"] == "add":
        check_dirs()
        if sys.platform == "win32":
            install_extension()
        else:
            if original_url == "" or flags["o"]:
                # Query GitHub API only if extension will be downloaded
                # from official GRASS addons repository
                get_addons_paths(gg_addons_base_dir=options["prefix"])
            source, url = resolve_source_code(
                name=options["extension"],
                url=original_url,
                branch=branch,
                fork=flags["o"],
            )
            xmlurl = resolve_xmlurl_prefix(original_url, source=source)
            install_extension(source=source, url=url, xmlurl=xmlurl, branch=branch)
    else:  # remove
        remove_extension(force=flags["f"])

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--doctest":
        import doctest

        sys.exit(doctest.testmod().failed)
    options, flags = gs.parser()
    global TMPDIR
    TMPDIR = tempfile.mkdtemp()
    atexit.register(cleanup)

    grass_version = gs.version()
    VERSION = grass_version["version"].split(".")

    sys.exit(main())
