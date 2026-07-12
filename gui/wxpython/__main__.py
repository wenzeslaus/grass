"""Launch the GRASS GUI with python -m grassgui

This requires a GRASS session, e.g., one created with grass --tmp-project,
or the grass gui subcommand which creates the session itself.
"""

import sys

from grassgui.wxgui import main

sys.exit(main())
