# RFC 18 Implementation Plan: Addon Management and Distribution

Status: Draft (implementation plan for
[RFC 18](RFC18_addon_management.md))

This plan breaks RFC 18 into thirteen ordered stages. Each stage is
independently shippable and roughly one reviewable pull request (stage
3 may split in two). Line numbers were verified against the actual
files as of this writing; they will drift as earlier stages land and
must be re-checked per stage.

Stages 1-4 are a behavior-preserving refactor guarded by the existing
*g.extension* test suites; the first deliberate behavior change (SVN
removal) is isolated in stage 2. The dependency track (stages 6-7),
the package track (stages 8-10), and the CLI/GUI track (stages 11-12)
are independent of each other once stage 5 lands.

## Stage overview

| Stage | Repo | Content | Depends on |
| --- | --- | --- | --- |
| 1 | grass | `grass.addons` skeleton, pure extraction, registry golden tests | -- |
| 2 | grass | fetch layer, `grass.utils.download` adoption, dead-code removal, wingrass HTTPS | 1 |
| 3 | grass | public verbs, *g.extension* and *g.extension.all* facades | 2 |
| 4 | grass | state hygiene, single clone, offline-clean local installs | 3 |
| 5 | grass | configurable endpoints and offline switch | 3 |
| 6 | both | manifest, dependency declare+check, `dependencies=` option, receipts | 3 |
| 7 | grass | dependency auto-install, addon-to-addon resolution | 6 |
| 8 | addons | `index.json` publication, noarch package CI, publish step | 6 (schema) |
| 9 | grass | client package channel, local package install, index cache | 5, 8 |
| 10 | addons | Windows binary packages per core release | 8 |
| 11 | grass | `grass addons` subcommand | 3-6 |
| 12 | grass | GUI migration to the library | 3-4 |
| 13 | addons | index becomes source of truth | 9 stable |

"addons" means the OSGeo/grass-addons repository; "both" means
coordinated changes in both repositories.

## Stage 1: `grass.addons` skeleton and pure extraction

New package `python/grass/addons/` with `__init__.py`,
`exceptions.py`, `reporter.py`, `config.py`, `naming.py`,
`resolve.py`, `registry.py`, `models.py`, and `tests/`.

Build-system wiring (three places):

- `python/grass/Makefile:8` -- add `addons` to `SUBDIRS`.
- `python/grass/CMakeLists.txt:1` -- add `addons` to `PYDIRS`.
- `python/grass/addons/Makefile` -- new, modeled on
  `python/grass/utils/Makefile`.

Functions migrated from `scripts/g.extension/g.extension.py` (moved
with globals turned into parameters; *g.extension* imports them and
deletes its copies):

| Into | From (g.extension.py) |
| --- | --- |
| `naming.py` | `expand_module_class_name` (596), `get_module_class_name` (626) -- both pure and doctested; doctests move along |
| `resolve.py` | `resolve_source_code` (2723), `KNOWN_HOST_SERVICES_INFO` (2612), `resolve_known_host_service` (2647), `validate_url` (2690), `resolve_xmlurl_prefix` (2586), `get_version_branch` (500), `get_default_branch` (528) |
| `config.py` | `resolve_install_prefix` (2549) minus the `os.environ` write and `gs.fatal`; constants `GIT_URL` (184), `HEADERS` (181), the make-program selection (186-193) |
| `registry.py` | writers `write_xml_modules` (1008), `write_xml_extensions` (1054), `write_xml_toolboxes` (1116) moved verbatim; readers `get_installed_extensions` (640), `get_installed_toolboxes` (665), `get_installed_modules` (688); mutators `install_extension_xml` (1374), `install_module_xml` (1498), `install_toolbox_xml` (1272), `remove_extension_xml` (2390), `remove_from_toolbox_xml` (2375) |

The SVN resolution branch is kept in this stage so the extraction is
strictly behavior-preserving.

Tests added (offline, pytest, in `python/grass/addons/tests/`):

- `grass_addons_registry_test.py` -- round-trips plus golden-file byte
  comparison of all three XML formats. The golden files are generated
  from the current code before the move, using fixtures copied from
  `scripts/g.extension/testsuite/data/`.
- `grass_addons_resolve_test.py` -- the source-type matrix (dir, zip,
  tar variants, known-host URLs, fork), largely converted from the
  existing doctests.

Open question resolved here: the source of the GRASS version and
libgis revision when no session exists (RFC 18 section 4.4). Candidate
answers to evaluate: build-time constants in
`python/grass/app/resource_paths.py` versus parsing the installed
`etc/VERSIONNUMBER` under `RuntimePaths().gisbase` versus requiring a
session only for verbs that need the values.

Exit criteria: all existing *g.extension* tests
(`scripts/g.extension/tests/`, `scripts/g.extension/testsuite/`) pass
unchanged; new unit tests pass without network or session.

## Stage 2: fetch layer, download consolidation, dead code

New `python/grass/addons/fetch.py`:

- `GitRepository` -- the reworked `GitAdapter`
  (`g.extension.py:198-457`): `gs.fatal` calls become `FetchError`,
  verbosity flags become the reporter, env is explicit.
- `fetch_source()` -- the dispatch of `download_source_code` (1858)
  minus the `svn` branch; the main-to-master retry and `fix_newlines`
  (1770) move along.

`python/grass/utils/download.py` extensions (backward-compatible):

- optional `headers=` on `download_and_extract` (187) and
  `_download_file` (149);
- optional explicit extraction directory;
- skip `__pycache__` entries in `extract_zip` (95), matching the one
  behavioral delta of *g.extension*'s fork (its `extract_zip` at 1799).

Deletions from *g.extension*: `urlretrieve` (483), `urlopen` (492),
`move_extracted_files` (1742), `extract_zip` (1799), `extract_tar`
(1825), `download_source_code_svn` (1673), the `svn` fallback in
`resolve_source_code` (2830) and its doctest, dead listing fallbacks
`list_available_extensions_svn` (917) and `get_wxgui_extensions`
(969).

Behavior changes, called out in release notes:

- SVN support is removed (done early, together with the stage 1 bug
  fixes): an unrecognized plain URL resolves to a plain-URL source
  type usable as a metadata location for listing, and installing from
  it produces a clear error instead of an attempted `svn checkout`.
- xz archives become supported (the `grass.utils.download` format list
  includes xz; *g.extension*'s own did not).
- The wingrass base URL (1603) switches to `https://`.

Tests: `grass_addons_fetch_test.py` with `file://` archive fixtures
(reusing `scripts/g.extension/testsuite/data/sample_modules/`) and a
local `git init` repository fixture mimicking the grass-addons `src/`
layout for `GitRepository` (branching, sparse checkout, path listing).

## Stage 3: public verbs and facades

New modules: `catalog.py`, `build.py`, `installer.py`, `windows.py`,
`removal.py`, `operations.py`; public API exported from `__init__.py`.

Migrations from `g.extension.py`:

| Into | From |
| --- | --- |
| `catalog.py` | `etree_fromfile` (565), `etree_fromurl` (570), listing internals `list_available_extensions` (730), `get_available_toolboxes` (757), `get_toolbox_extensions` (783), `get_module_files` (813), `get_module_executables` (828), `get_optional_params` (847), `list_available_modules` (868), `get_toolboxes_metadata` (1240), `get_addons_metadata` (1321) -- rewritten to return data; formatting stays in the tool |
| `build.py` | make and CMake command construction and tool-name scanning from `install_extension_std_platforms` (1967; make 2108-2156, cmake 2050-2107, scanning 1998-2032); `check_progs` (585) becomes `check_prerequisites()` |
| `installer.py` | orchestration from `install_extension` (1151) and `install_extension_std_platforms` (1967); `check_dirs` (2466), `create_dir` (2450), `check_style_file` (2424); shebang rewrite (2034-2048); `create_md_if_missing` (1950); `update_manual_page` (2484); multi-addon helpers (1449, 1473) |
| `windows.py` | `install_extension_win` (1595), `replace_shebang_win` (460) |
| `removal.py` | `remove_extension` (2191), `remove_extension_files` (2296), `remove_extension_std` (2348) with dry-run returning the file list |

Facade rewrite of `scripts/g.extension/g.extension.py`: parser header
(24-142) unchanged; `main()` (2877) becomes the option-to-verb mapping
from RFC 18 section 4.6 plus existing output formatting; target size
300-400 lines.

*g.extension.all* (`scripts/g.extension.all/g.extension.all.py`):
`get_extensions` (58) is replaced by `grass.addons.outdated()`;
`find_addon_name` (155) and the duplicate downloader (`urlopen` 87,
`download_modules_xml_file` 95) are replaced by `catalog` calls; the
per-addon subprocess loop in `main` (203) stays.

Tests: `grass_addons_catalog_test.py` (`file://` XML fixtures),
`grass_addons_removal_test.py` (synthetic prefix trees),
`grass_addons_install_test.py` (end-to-end install from a local
directory with a minimal script-only addon fixture; requires a session
and make, marked accordingly). The GUI is deliberately untouched and
serves as an integration check of facade fidelity.

## Stage 4: state hygiene, single clone, offline guarantee

- Remove the `os.environ` writes (`GRASS_PREFIX_ADDON_BASE` at 2580,
  `GRASS_PROXY` at 2895 in the pre-refactor numbering); both values go
  only into explicitly built subprocess environments in `build.py`.
- Replace the global urllib opener (2893) with a per-request opener
  built from `config.proxies`.
- Replace `TMPDIR`/`atexit`/`REMOVE_TMPDIR` and all `os.chdir` calls
  with a per-operation workdir context manager honoring
  `download_only`/`build_only`.
- One `GitRepository` per operation: the extra blobless clones from
  `get_addons_paths` (2833) -- reached from `main()`, from
  `filter_multi_addon_addons` (1473), and from `update_manual_page`
  (2484) -- become method calls on the already-open repository.
- Thread `check_cancelled()` through fetch, build, and file-walk
  loops.

Tests: an offline test asserting that installing from a local
directory and from a local zip performs zero network operations
(socket-blocking fixture), completing what commit `740266b459`
started; a cancellation test asserting workdir cleanup.

## Stage 5: configurable endpoints

- `AddonsConfig` fields for the official repository URL, catalog URL,
  Windows binary server URL, and package repository URL become
  user-reachable: environment variables `GRASS_ADDON_REPOSITORY` and
  `GRASS_ADDON_OFFLINE`, read in `default_config()`.
- The Windows path (`windows.py`) honors the configured server instead
  of the hardcoded one.
- `GRASS_ADDON_OFFLINE=1` short-circuits all network fallbacks in the
  library, and gates the GitHub API fallback in documentation
  generation: one conditional in `utils/mkdocs.py:get_last_git_commit`
  (130) so that `get_git_commit_from_rest_api_for_addon_repo` (215) is
  skipped and the existing fallback (`get_git_commit_from_file`, 259)
  is used.

## Stage 6: manifest and dependency declare+check

Core repository:

- `manifest.py` -- `addon.toml` reading (`tomllib`; Python floor is
  3.11 per `pyproject.toml:3`) and validation; the schema of RFC 18
  section 5.2 frozen jointly with the grass-addons side.
- `dependencies.py` -- installed-distribution map via
  `importlib.metadata.distributions()` with PEP 503 normalization;
  two-tier requirement parsing (`packaging` when importable,
  presence-only fallback); environment detection returning
  `PythonEnvironment` (RFC 18 section 6.3); report rendering;
  `requirements.txt` fallback parser.
- `dependencies=check|install|require|none` option (answer `check`)
  added to the *g.extension* parser header; `check` and `require`
  wired into `operations.install` after fetch and before build;
  `install` values accepted but deferred to stage 7 (documented).
- Receipts: `<prefix>/.registry/<addon>/manifest.toml` (or
  `manifest.json` for the recovered-requirements case) and
  `install.json` written by `registry.py`; XML output remains pinned
  by the golden tests.

grass-addons repository (proposal, coordinated PR):

- `addon.toml` added to roughly ten pilot addons.
- A CI lint flagging addons with a `requirements.txt` but no manifest
  dependency block; contributor documentation for the migration.

Tests: fake-environment fixtures (a `pyvenv.cfg` with a `uv` key, a
`conda-meta/` directory, an `EXTERNALLY-MANAGED` marker under a stub
`sysconfig` path) driving detection; manifest validation; report
golden text; a `require` failure test.

## Stage 7: auto-install and addon-to-addon resolution

- Executors per detected environment (pip via `sys.executable`, uv,
  pip-in-conda, pip on OSGeo4W), command echo before execution, PEP
  668 refusal, no-sudo and no-`--break-system-packages` invariants,
  proceed-with-warning on installer failure.
- `resolve_addon_order()` (DFS post-order, cycle error with path),
  satisfied-by-core-tool check, aggregation of Python requirements
  across the resolved set into one report and one installer run,
  `dependent_addons()` reverse scan of stored manifests wired into the
  removal preview.

Tests: executor selection against the fake environments (asserting the
exact argv, never executing a real installer); resolution order,
cycles, and already-installed skipping over synthetic manifests.

## Stage 8: index and noarch packages (OSGeo/grass-addons)

- `utils/packaging/` -- package builder (zip of a scratch install
  prefix plus manifest with the appended `[build]` table), manifest
  synthesizer (from `--interface-description` output and git history,
  including `0.<YYYYMMDD>` version synthesis), and `index.json`
  generator covering every addon in the branch.
- `.github/workflows/package-addons.yml` -- installs GRASS and builds
  per-addon scratch prefixes following the pattern proven in the core
  repository's `documentation.yml` (grass-addons checkout at
  `.github/workflows/documentation.yml:77`, per-addon build via
  `compile_addons_git.sh` at `:185`); packages a build only when the
  prefix contains no compiled objects (noarch); uploads the
  `packages/` tree as one artifact.
- One new step in `utils/addons/grass-addons-publish.sh` (the script
  that already generates `modules.xml`; see `doc/infrastructure.md:220`
  in the core repository): pull the latest successful artifact
  (`gh run download`), verify, rsync with `index.json` last.
- First sub-deliverable is the index alone (all addons, empty
  `versions`), which is a pure addition next to `modules.xml`.

## Stage 9: client package channel

- `PrebuiltPackageBuilder` (no build; verify sha256, extract, merge)
  behind the `Builder` protocol; resolution chain of RFC 18 section
  5.4 in `operations.install` with the single info-level fallback
  message.
- Local package zips (detected by `addon.toml` at the archive root)
  installable with zero network; `[build].libgis_revision` copied into
  the local `modules.xml`.
- Index cache: one file per repository under
  `$GRASS_ADDON_BASE/cache/` with conditional GET and a dated warning
  when used offline.

Tests: a fixture package tree with `index.json` served over `file://`;
checksum-failure, platform-selection, and fallback-chain tests; an
offline package install test under the socket-blocking fixture.

## Stage 10: Windows binary packages (OSGeo/grass-addons)

OSGeo4W-based addon builds on Windows runners, triggered per core
release, populating `windows-x86_64/grass-X.Y.Z/` in the package tree.
The CTU wingrass pipeline and `landam/wingrass-maintenance-scripts`
remain untouched as the parallel fallback. This is the highest-risk
stage (toolchain drift, OSGeo4W CI flakiness) and nothing else depends
on it.

## Stage 11: `grass addons` subcommand

- `add_addons_subparser()` in `python/grass/app/cli.py` following the
  noun-verb pattern of `add_mapset_subparser` (94) and
  `add_project_subparser` (193), registered in `main` (235); verbs
  `install`, `remove`, `list`, `search`, `check`, `upgrade` per RFC 18
  section 7.2, `--format {plain,json}` on the read-only verbs.
- Add `"addons"` to the launcher dispatch set at
  `lib/init/grass.py:2036`.
- Sessionless operation via the stage 1 decision; plain-stderr
  reporter; `check` exits nonzero on missing required dependencies.

Tests: extend the pattern of
`python/grass/app/tests/grass_app_cli_test.py` (direct `main()` calls
plus `python -m grass.app` subprocess) with the new verbs and JSON
output.

## Stage 12: GUI migration

`gui/wxpython/modules/extensions.py` switches from subprocess calls
(listing at 361 and 549, install at 532, removal at 486 and 509) to
library verbs on a worker thread: a reporter marshaling through
`wx.CallAfter`, cancellation via a `threading.Event` checked in
`check_cancelled`, `RemovalResult.files` feeding the confirmation
dialog, and `DependencyReport` rendered with the install-packages
button disabled when the environment refuses auto-install. wxPython
conventions apply in this file per the project guidelines.

## Stage 13: index as source of truth (OSGeo/grass-addons)

After stage 9 has been stable for a release cycle: `modules.xml` and
`toolboxes.xml` become derived outputs of the index generator
(byte-compatible), and client listings prefer the index with
`modules.xml` as fallback for servers that only publish the legacy
files. Legacy source channels remain supported indefinitely.

## Risks

- **Facade fidelity** (stages 3-4): mitigated by keeping the existing
  *g.extension* pytest and gunittest suites green at every stage, by
  golden-file registry tests, and by the GUI continuing to exercise
  the tool's CLI until stage 12.
- **Registry byte drift**: any deviation in the three XML files breaks
  *g.search.modules*, the GUI, and *g.extension.all*; the writers move
  verbatim and the golden tests fail on any change.
- **Windows CI builds** (stage 10): may prove flaky or drift from the
  CTU toolchain; the wingrass fallback stays in the resolution chain,
  so the stage can slip without blocking anything else.
- **Environment misdetection** (stages 6-7): bounded by the
  echo-before-run rule and the refusal path -- the worst case is a
  wrong printed suggestion, never a silently mutated environment.
- **Partial package coverage** (stages 8-9): handled by design; the
  index lists all addons and unpackaged ones fall back to the legacy
  channel with one info message.
- **Line-number drift in this plan**: numbers reference the
  pre-refactor files and shift as stages land; each stage PR
  re-verifies the references it touches.
