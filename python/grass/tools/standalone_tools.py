##############################################################################
# AUTHOR(S): Vaclav Petras <wenzeslaus gmail com>
#
# PURPOSE:   API to call GRASS tools as Python functions without a session
#
# SPDX-FileCopyrightText: 2025-2026 Vaclav Petras
# SPDX-FileCopyrightText: GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

"""
An API to call GRASS tools as Python functions without a session

This is not a stable part of the API. Use at your own risk.
"""

import os
import tempfile
import json
import subprocess
import weakref
from pathlib import Path

import grass.script as gs
from grass.grassdb.create import set_project_crs_from_pack
from .importexport import ImporterExporter
from .session_tools import Tools
from .support import ParameterConverter, ToolFunctionResolver


class StandaloneTools:
    """Use GRASS tools through function calls without a session (experimental)

    On first use, the object creates a temporary XY project and a session in it.
    The first native raster file (raster pack) given as an input sets the CRS of
    the project and, unless the region was set with *g.region*, the computational
    region. Inputs are imported with *r.unpack* and outputs are written with
    *r.pack*.

    .. code-block:: python

        tools = StandaloneTools()
        tools.r_slope_aspect(elevation="dem.grass_raster", slope="slope.grass_raster")

    This class is experimental and its behavior may change.
    """

    def __init__(
        self, session=None, env=None, work_dir=None, errors=None, capture_output=True
    ):
        self._tools = None
        self._errors = errors
        self._capture_output = capture_output
        self._crs_initialized = False
        self._work_dir = work_dir
        self._tmp_dir = None
        self._tmp_dir_finalizer = None
        if session and env:
            msg = "Provide either session or env, not both"
            raise ValueError(msg)
        self._session = None
        if session:
            # If session is provided, we will use it as is.
            self._session = session
            self._crs_initialized = True
            self._original_env = None  # env should not be used
        elif env:
            self._original_env = env
        else:
            self._original_env = os.environ
        self._region_is_set = False
        self._project_path = None
        self._region_file = None
        self._region_modified_time = None
        self._name_helper = None

    def _process_parameters(self, command, popen_options):
        if not self._session:
            # We create session and an empty XY project in one step.
            self._create_session()

        env = popen_options.get("env", self._session.env)

        # Start the tool as grass.script does, so that it is found on the PATH
        # of the session and a tool which is a script runs on MS Windows, too.
        process = gs.Popen(
            [*command, "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(
            process.args, process.returncode, stdout=stdout, stderr=stderr
        )

    def run(self, name, /, **kwargs):
        object_parameter_handler = ParameterConverter()
        object_parameter_handler.process_parameters(kwargs)

        args, popen_options = gs.popen_args_command(name, **kwargs)

        interface_result = self._process_parameters(args, popen_options)
        if interface_result.returncode != 0:
            # This is only for the error states.
            return gs.handle_errors(
                interface_result.returncode,
                result=None,
                args=[name],
                kwargs=kwargs,
                stderr=interface_result.stderr,
                handler="raise",
            )
        parameters = json.loads(interface_result.stdout)

        if not self._session:
            # We create session and an empty XY project in one step.
            self._create_session()

        rows_columns = object_parameter_handler.input_rows_columns()
        if not self._is_region_modified() and rows_columns:
            rows, cols = rows_columns
            self.no_nonsense_run(
                "g.region",
                rows=rows,
                cols=cols,
                env=self._session.env,
            )
            self._region_is_set = True

        object_parameter_handler.translate_objects_to_data(
            kwargs, env=self._session.env
        )

        # We approximate tool_kwargs as original kwargs.
        result = self.run_from_list(
            args,
            tool_kwargs=kwargs,
            processed_parameters=parameters,
            stdin=object_parameter_handler.stdin,
            **popen_options,
        )
        use_objects = object_parameter_handler.translate_data_to_objects(
            kwargs, env=self._session.env
        )
        if use_objects:
            result = object_parameter_handler.result
        return result

    def run_from_list(
        self,
        command,
        tool_kwargs=None,
        stdin=None,
        processed_parameters=None,
        **popen_options,
    ):
        """Run a tool given as a list of strings (--help is not supported)"""
        if not self._session:
            # We create session and an empty XY project in one step.
            self._create_session()

        pack_importer_exporter = ImporterExporter(
            run_function=self.no_nonsense_run,
            run_cmd_function=self.no_nonsense_run_from_list,
        )
        command = pack_importer_exporter.process_parameter_list(command)

        if not self._crs_initialized:
            # The rasters are in the list of current ones until they are imported.
            self._initialize_crs(pack_importer_exporter.current_input_rasters)

        pack_importer_exporter.import_data(env=self._session.env)

        if not self._is_region_modified() and pack_importer_exporter.input_rasters:
            self.no_nonsense_run(
                "g.region",
                raster=pack_importer_exporter.input_rasters[0][1],
                env=self._session.env,
            )
            self._region_is_set = True
        result = self.no_nonsense_run_from_list(command)
        # An existing file is replaced as existing data in the project is.
        pack_importer_exporter.export_data(env=self._session.env, delete_first=True)
        return result

    def no_nonsense_run(self, name, /, *, tool_kwargs=None, stdin=None, **kwargs):
        args, popen_options = gs.popen_args_command(name, **kwargs)
        return self.no_nonsense_run_from_list(
            args, tool_kwargs=tool_kwargs, stdin=stdin, **popen_options
        )

    def no_nonsense_run_from_list(
        self, command, tool_kwargs=None, stdin=None, **popen_options
    ):
        if not self._session:
            # We create session and an empty XY project in one step.
            self._create_session()
        if not self._tools:
            self._tools = Tools(
                overwrite=True,
                quiet=False,
                verbose=False,
                superquiet=False,
                errors=self._errors,
                capture_output=self._capture_output,
                session=self._session,
            )
        return self._tools.call_cmd(command)

    def _create_session(self):
        # Temporary folder for all our files
        if self._work_dir:
            base_dir = self._work_dir
        else:
            # Resource is managed by weakref.finalize.
            self._tmp_dir = (
                # pylint: disable=consider-using-with
                tempfile.TemporaryDirectory()
            )

            def cleanup(tmpdir):
                tmpdir.cleanup()

            self._tmp_dir_finalizer = weakref.finalize(self, cleanup, self._tmp_dir)
            base_dir = self._tmp_dir.name
        project_name = "project"
        project_path = Path(base_dir) / project_name
        gs.create_project(project_path)
        self._project_path = project_path
        self._region_file = project_path / "PERMANENT" / "WIND"
        self._region_modified_time = self._region_file.stat().st_mtime
        self._session = gs.setup.init(project_path, env=self._original_env.copy())

    def _initialize_crs(self, rasters):
        if not rasters:
            return
        # Setting the CRS resets the computational region, so we keep the region
        # the user may have set, except for the first two lines of the file which
        # have the CRS type and zone.
        user_region = None
        if self._is_region_modified():
            user_region = self._region_file.read_text().splitlines()[2:]
        set_project_crs_from_pack(self._project_path, rasters[0][0])
        if user_region:
            crs_lines = self._region_file.read_text().splitlines()[:2]
            self._region_file.write_text("\n".join([*crs_lines, *user_region]) + "\n")
        else:
            self._region_modified_time = self._region_file.stat().st_mtime
        # Later inputs are checked against this CRS when they are imported.
        self._crs_initialized = True

    def _is_region_modified(self):
        if self._region_is_set:
            return True
        if not self._region_file:
            if self._session:
                self._region_file = (
                    Path(
                        self.no_nonsense_run(
                            "g.gisenv", get="GISDBASE,LOCATION_NAME", sep="/"
                        ).text
                    )
                    / "PERMANENT"
                    / "WIND"
                )
                self._region_modified_time = self._region_file.stat().st_mtime
            return False
        return self._region_file.stat().st_mtime > self._region_modified_time

    def cleanup(self):
        if self._tmp_dir_finalizer:
            self._tmp_dir_finalizer()

    def __enter__(self):
        """Enter the context manager context.

        :returns: reference to the object (self)
        """
        return self

    def __exit__(self, type, value, traceback):
        """Exit the context manager context.

        Removes the temporary project, see :meth:`cleanup`.
        """
        self.cleanup()

    def __getattr__(self, name):
        """Return a function which runs the tool, e.g., r_slope_aspect for r.slope.aspect"""
        if not self._session:
            # We create session and an empty XY project in one step.
            self._create_session()
        if not self._name_helper:
            self._name_helper = ToolFunctionResolver(
                run_function=self.run, env=self._session.env
            )
        return self._name_helper.get_function(name, exception_type=AttributeError)
