# MODULE:    grass.addons
#
# AUTHOR(S): Vaclav Petras <wenzeslaus gmail com>
#
# PURPOSE:   Addon (extension) management library
#
# COPYRIGHT: (C) 2026 by Vaclav Petras, and the GRASS Development Team
#
#            This program is free software under the GNU General Public
#            License (>=v2). Read the file COPYING that comes with GRASS
#            for details.

"""Progress and message reporting for the grass.addons package

Functions in the package take an optional *reporter* object and send
all user-facing messages through it instead of printing. Callers
provide an implementation which fits their interface, for example one
which forwards to the messaging functions of grass.script in a tool or
one which updates a GUI. When no reporter is given, functions use
:class:`NullReporter` and stay silent.

A reporter is also the cancellation channel: long-running operations
call :meth:`Reporter.check_cancelled` between steps, and an
implementation may raise
:class:`~grass.addons.exceptions.OperationCancelled` there to stop the
operation cooperatively.
"""

from typing import Protocol


class Reporter(Protocol):
    """Interface for receiving messages from grass.addons operations"""

    def message(self, text: str) -> None:
        """Report a standard informational message"""

    def verbose(self, text: str) -> None:
        """Report a message shown only in verbose mode"""

    def debug(self, text: str) -> None:
        """Report a debugging message"""

    def warning(self, text: str) -> None:
        """Report a warning"""

    def check_cancelled(self) -> None:
        """Raise OperationCancelled if the operation should stop"""


class NullReporter:
    """Reporter which discards all messages and never cancels"""

    def message(self, text: str) -> None:
        pass

    def verbose(self, text: str) -> None:
        pass

    def debug(self, text: str) -> None:
        pass

    def warning(self, text: str) -> None:
        pass

    def check_cancelled(self) -> None:
        pass
