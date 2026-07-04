"""grassgui - the wxPython-based GRASS graphical user interface

This is not a stable API. Use the grass package for scripting.
"""

import os as _os

# In the legacy (Autotools-compatible) layout, this file is installed on
# the Python path as a stand-in package while the package content lives
# in the wxGUI directory under GISBASE. Point the package there in that
# case; when the content is next to this file (FHS layout), do nothing.
if not _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "core")):
    _guiwx = _os.environ.get("GRASS_GUIWXDIR")
    if _guiwx and _os.path.isdir(_guiwx):
        __path__ = [_guiwx]
