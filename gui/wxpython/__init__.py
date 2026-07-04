"""grassgui - the wxPython-based GRASS graphical user interface

This is not a stable API. Use the grass package for scripting.
"""

import os as _os

# In the legacy (Autotools-compatible) layout, this file is installed on
# the Python path as a stand-in package while the package content lives
# in the wxGUI directory under GISBASE. Point the package there in that
# case; when the content is next to this file (FHS layout), do nothing.
# The relative fallback covers environments without session variables
# (the stand-in is at <gisbase>/etc/python/grassgui/__init__.py).
if not _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "core")):
    _guiwx = _os.environ.get("GRASS_GUIWXDIR")
    if not _guiwx or not _os.path.isdir(_guiwx):
        _base = _os.path.dirname(  # gisbase
            _os.path.dirname(  # etc
                _os.path.dirname(  # python
                    _os.path.dirname(_os.path.abspath(__file__))  # grassgui
                )
            )
        )
        _guiwx = _os.path.join(_base, "gui", "wxpython")
    if _os.path.isdir(_guiwx):
        __path__ = [_guiwx]
