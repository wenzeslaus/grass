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

"""Local metadata files about installed addons"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.parsers import expat

from grass.script import task as gtask

from .reporter import NullReporter

# XML parsing exceptions to catch when reading the metadata files.
ETREE_EXCEPTIONS = (ET.ParseError, expat.ExpatError)


def etree_fromfile(filename):
    """Create XML element tree from a given file name"""
    return ET.fromstring(Path(filename).read_text(encoding="utf-8"))


def get_module_files(mnode):
    """Return list of module files

    :param mnode: XML node for a module
    """
    flist = []
    if mnode.find("binary") is None:
        return flist
    for file_node in mnode.find("binary").findall("file"):
        filepath = file_node.text
        flist.append(filepath)

    return flist


def get_module_executables(mnode, prefix):
    """Return list of module executables

    :param mnode: XML node for a module
    :param prefix: addon installation prefix used to recognize executables
    """
    flist = []
    for filepath in get_module_files(mnode):
        if filepath.startswith(prefix + os.path.sep + "bin") or (
            sys.platform != "win32"
            and filepath.startswith(prefix + os.path.sep + "scripts")
        ):
            filename = os.path.basename(filepath)
            if sys.platform == "win32":
                filename = os.path.splitext(filename)[0]
            flist.append(filename)

    return flist


def get_optional_params(mnode):
    """Return description and keywords of a module as a tuple

    :param mnode: XML node for a module
    """
    try:
        desc = mnode.find("description").text
    except AttributeError:
        desc = ""
    if desc is None:
        desc = ""
    try:
        keyw = mnode.find("keywords").text
    except AttributeError:
        keyw = ""
    if keyw is None:
        keyw = ""

    return desc, keyw


class LocalRegistry:
    """Metadata files about addons installed under one prefix

    The files modules.xml, extensions.xml, and toolboxes.xml directly
    under *prefix* record which addons are installed there.
    """

    def __init__(self, prefix, *, version_major, libgis_revision, reporter=None):
        """
        :param prefix: addon installation prefix containing the metadata files
        :param version_major: major GRASS version recorded in the files
        :param libgis_revision: GIS library revision recorded in the files
        :param reporter: object receiving messages (defaults to NullReporter)
        """
        if reporter is None:
            reporter = NullReporter()
        self.prefix = prefix
        self.version_major = version_major
        self.libgis_revision = libgis_revision
        self._reporter = reporter

    def _prefixed_path(self, path):
        """Return the file path anchored under the installation prefix

        Paths which are already absolute or already lie under the prefix
        are returned unchanged, so re-writing a previously written file
        does not prepend the prefix a second time.

        :param path: file path from a metadata file entry
        """
        if os.path.isabs(path) or Path(path).is_relative_to(self.prefix):
            return path
        return os.path.join(self.prefix, path)

    def write_xml_modules(self, name, tree=None):
        """Write element tree as a modules metadata file

        If the *tree* is not given, an empty file is created.

        :param name: file name
        :param tree: XML element tree
        """
        file_ = open(name, "w", encoding="utf-8")  # noqa: SIM115
        file_.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file_.write('<!DOCTYPE task SYSTEM "grass-addons.dtd">\n')
        file_.write(f'<addons version="{self.version_major}">\n')

        if tree is not None:
            for tnode in tree.findall("task"):
                indent = 4
                file_.write('%s<task name="%s">\n' % (" " * indent, tnode.get("name")))
                indent += 4
                file_.write(
                    "%s<description>%s</description>\n"
                    % (" " * indent, tnode.find("description").text or "")
                )
                file_.write(
                    "%s<keywords>%s</keywords>\n"
                    % (" " * indent, tnode.find("keywords").text or "")
                )
                bnode = tnode.find("binary")
                if bnode is not None:
                    file_.write("%s<binary>\n" % (" " * indent))
                    indent += 4
                    file_.writelines(
                        "%s<file>%s</file>\n"
                        % (" " * indent, self._prefixed_path(fnode.text))
                        for fnode in bnode.findall("file")
                    )
                    indent -= 4
                    file_.write("%s</binary>\n" % (" " * indent))
                file_.write(
                    '%s<libgis revision="%s" />\n'
                    % (" " * indent, self.libgis_revision)
                )
                indent -= 4
                file_.write("%s</task>\n" % (" " * indent))

        file_.write("</addons>\n")
        file_.close()

    def write_xml_extensions(self, name, tree=None):
        """Write element tree as an extensions metadata file

        If the *tree* is not given, an empty file is created.

        :param name: file name
        :param tree: XML element tree
        """
        file_ = open(name, "w", encoding="utf-8")  # noqa: SIM115
        file_.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file_.write('<!DOCTYPE task SYSTEM "grass-addons.dtd">\n')
        file_.write(f'<addons version="{self.version_major}">\n')

        if tree is not None:
            for tnode in tree.findall("task"):
                indent = 4
                # extension name
                file_.write('%s<task name="%s">\n' % (" " * indent, tnode.get("name")))
                indent += 4

                # extension files
                bnode = tnode.find("binary")
                if bnode is not None:
                    file_.write("%s<binary>\n" % (" " * indent))
                    indent += 4
                    file_.writelines(
                        "%s<file>%s</file>\n"
                        % (" " * indent, self._prefixed_path(fnode.text))
                        for fnode in bnode.findall("file")
                    )
                    indent -= 4
                    file_.write("%s</binary>\n" % (" " * indent))
                # extension modules
                mnode = tnode.find("modules")
                if mnode is not None:
                    file_.write("%s<modules>\n" % (" " * indent))
                    indent += 4
                    file_.writelines(
                        "%s<module>%s</module>\n" % (" " * indent, fnode.text)
                        for fnode in mnode.findall("module")
                    )
                    indent -= 4
                    file_.write("%s</modules>\n" % (" " * indent))

                file_.write(
                    '%s<libgis revision="%s" />\n'
                    % (" " * indent, self.libgis_revision)
                )
                indent -= 4
                file_.write("%s</task>\n" % (" " * indent))

        file_.write("</addons>\n")
        file_.close()

    def write_xml_toolboxes(self, name, tree=None):
        """Write element tree as a toolboxes metadata file

        If the *tree* is not given, an empty file is created.

        :param name: file name
        :param tree: XML element tree
        """
        file_ = open(name, "w", encoding="utf-8")  # noqa: SIM115
        file_.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file_.write('<!DOCTYPE toolbox SYSTEM "grass-addons.dtd">\n')
        file_.write(f'<addons version="{self.version_major}">\n')
        if tree is not None:
            for tnode in tree.findall("toolbox"):
                indent = 4
                file_.write(
                    '%s<toolbox name="%s" code="%s">\n'
                    % (" " * indent, tnode.get("name"), tnode.get("code"))
                )
                indent += 4
                file_.writelines(
                    '%s<correlate code="%s" />\n' % (" " * indent, cnode.get("code"))
                    for cnode in tnode.findall("correlate")
                )
                file_.writelines(
                    '%s<task name="%s" />\n' % (" " * indent, mnode.get("name"))
                    for mnode in tnode.findall("task")
                )
                indent -= 4
                file_.write("%s</toolbox>\n" % (" " * indent))

        file_.write("</addons>\n")
        file_.close()

    def get_installed_toolboxes(self, force=False):
        """Get list of installed toolboxes

        Writes toolboxes file if it does not exist and *force* is set to
        ``True``. Creates a new toolboxes file if it is not possible
        to read the current one.
        """
        xml_file = os.path.join(self.prefix, "toolboxes.xml")
        if not Path(xml_file).exists():
            if force:
                self.write_xml_toolboxes(xml_file)
            else:
                self._reporter.debug("No addons metadata file available")
            return []
        # read XML file
        try:
            tree = etree_fromfile(xml_file)
        except ETREE_EXCEPTIONS + (OSError,):
            os.remove(xml_file)
            self.write_xml_toolboxes(xml_file)
            return []
        ret = []
        for tnode in tree.findall("toolbox"):
            ret.append(tnode.get("code"))
        return ret

    def get_installed_modules(self, force=False, shell_format=False):
        """Get list of installed modules.

        Writes modules file if it does not exist and *force* is set to ``True``.
        Creates a new modules file if it is not possible
        to read the current one.
        """
        xml_file = os.path.join(self.prefix, "modules.xml")
        if not Path(xml_file).exists():
            if force:
                self.write_xml_modules(xml_file)
            else:
                self._reporter.debug("No addons metadata file available")
            return []
        # read XML file
        try:
            tree = etree_fromfile(xml_file)
        except ETREE_EXCEPTIONS + (OSError,):
            os.remove(xml_file)
            self.write_xml_modules(xml_file)
            return []
        ret = []
        for tnode in tree.findall("task"):
            if shell_format:
                desc, keyw = get_optional_params(tnode)
                ret.extend(
                    (
                        "name={0}".format(tnode.get("name").strip()),
                        "description={0}".format(desc),
                        "keywords={0}".format(keyw),
                        "executables={0}".format(
                            ",".join(get_module_executables(tnode, self.prefix))
                        ),
                    )
                )
            else:
                ret.append(tnode.get("name").strip())

        return ret

    def install_toolbox_xml(self, code, tdata):
        """Update local toolboxes metadata file

        :param code: toolbox code
        :param tdata: toolbox metadata with name, correlate, and modules keys
        """
        xml_file = os.path.join(self.prefix, "toolboxes.xml")
        # create an empty file if not exists
        if not Path(xml_file).exists():
            self.write_xml_toolboxes(xml_file)

        # read XML file
        tree = etree_fromfile(xml_file)

        # update tree
        tnode = None
        for node in tree.findall("toolbox"):
            if node.get("code") == code:
                tnode = node
                break

        if tnode is not None:
            # update existing node
            for cnode in tnode.findall("correlate"):
                tnode.remove(cnode)
            for mnode in tnode.findall("task"):
                tnode.remove(mnode)
        else:
            # create new node for task
            tnode = ET.Element("toolbox", attrib={"name": tdata["name"], "code": code})
            tree.append(tnode)

        for cname in tdata["correlate"]:
            cnode = ET.Element("correlate", attrib={"code": cname})
            tnode.append(cnode)
        for tname in tdata["modules"]:
            mnode = ET.Element("task", attrib={"name": tname})
            tnode.append(mnode)

        self.write_xml_toolboxes(xml_file, tree)

    def install_extension_xml(self, edict):
        """Update XML files with metadata about installed modules and toolbox
        of a private addon
        """
        xml_file = os.path.join(self.prefix, "extensions.xml")
        # create an empty file if not exists
        if not Path(xml_file).exists():
            self.write_xml_extensions(xml_file)

        # read XML file
        tree = etree_fromfile(xml_file)

        # update tree
        for name in edict:
            # so far extensions do not have description or keywords
            # only modules have

            tnode = None
            for node in tree.findall("task"):
                if node.get("name") == name:
                    tnode = node
                    break

            if tnode is None:
                # create new node for task
                tnode = ET.Element("task", attrib={"name": name})

                # create binary
                bnode = ET.Element("binary")
                # list of all installed files for this extension
                for file_name in edict[name]["flist"]:
                    fnode = ET.Element("file")
                    fnode.text = file_name
                    bnode.append(fnode)
                tnode.append(bnode)

                # create modules
                msnode = ET.Element("modules")
                # list of all installed modules for this extension
                for module_name in edict[name]["mlist"]:
                    mnode = ET.Element("module")
                    mnode.text = module_name
                    msnode.append(mnode)
                tnode.append(msnode)
                tree.append(tnode)
            else:
                self._reporter.verbose(
                    "Extension already listed in metadata file; metadata not updated!"
                )
        self.write_xml_extensions(xml_file, tree)

    def install_module_xml(self, mlist):
        """Update XML files with metadata about installed modules and toolbox
        of a private addon
        """
        xml_file = os.path.join(self.prefix, "modules.xml")
        # create an empty file if not exists
        if not Path(xml_file).exists():
            self.write_xml_modules(xml_file)

        # read XML file
        tree = etree_fromfile(xml_file)

        # update tree
        for name in mlist:
            try:
                desc = gtask.parse_interface(name).description
                keywords = gtask.parse_interface(name).keywords
            except Exception as error:
                self._reporter.warning(
                    _("No metadata available for module '{name}': {error}").format(
                        name=name, error=error
                    )
                )
                continue

            tnode = None
            for node in tree.findall("task"):
                if node.get("name") == name:
                    tnode = node
                    break

            if tnode is None:
                # create new node for task
                tnode = ET.Element("task", attrib={"name": name})
                dnode = ET.Element("description")
                dnode.text = desc
                tnode.append(dnode)
                knode = ET.Element("keywords")
                knode.text = (",").join(keywords)
                tnode.append(knode)

                # binary files installed with an extension are now
                # listed in extensions.xml
                tree.append(tnode)
            else:
                self._reporter.verbose(
                    "Extension module already listed in metadata file; metadata not "
                    "updated!"
                )
        self.write_xml_modules(xml_file, tree)

    def remove_from_toolbox_xml(self, name):
        """Update local meta-file when removing existing toolbox"""
        xml_file = os.path.join(self.prefix, "toolboxes.xml")
        if not Path(xml_file).exists():
            return
        # read XML file
        tree = etree_fromfile(xml_file)
        for node in tree.findall("toolbox"):
            if node.get("code") != name:
                continue
            tree.remove(node)

        self.write_xml_toolboxes(xml_file, tree)

    def remove_extension_xml(self, mlist, edict, extension_name):
        """Update local meta-file when removing existing extension

        :param mlist: names of the removed modules
        :param edict: metadata of the removed extensions
        :param extension_name: name of the removed extension (used to update
            the toolboxes file when the extension has multiple entries)
        """
        if len(edict) > 1:
            # update also toolboxes metadata
            self.remove_from_toolbox_xml(extension_name)

        # modules
        xml_file = os.path.join(self.prefix, "modules.xml")
        if Path(xml_file).exists():
            # read XML file
            tree = etree_fromfile(xml_file)
            for name in mlist:
                for node in tree.findall("task"):
                    if node.get("name") != name:
                        continue
                    tree.remove(node)
            self.write_xml_modules(xml_file, tree)

        # extensions
        xml_file = os.path.join(self.prefix, "extensions.xml")
        if Path(xml_file).exists():
            # read XML file
            tree = etree_fromfile(xml_file)
            for name in edict:
                for node in tree.findall("task"):
                    if node.get("name") != name:
                        continue
                    tree.remove(node)
            self.write_xml_extensions(xml_file, tree)
