# FHS implementation plan (local working document)

Plan for completing Filesystem Hierarchy Standard (FHS) support in the
CMake build. CMake must support both the current Autotools-compatible
layout (`WITH_FHS=OFF`, the default) and the new FHS layout
(`WITH_FHS=ON`). Autotools gets no FHS support (it will be dropped
eventually) and must not regress.

This file tracks local work on the `fhs-implementation` branch; it is not
intended for a PR.

## Status (as of 2026-07-04)

- CMake already has the `WITH_FHS` option with dual layouts defined in
  `cmake/modules/GRASSInstallDirs.cmake` (legacy: everything under
  `lib/grass<MM>/` = `GISBASE`; FHS: `libexec/grass/` for binaries, plain
  `libdir` for libraries, `site-packages` for Python, `share/grass/` for
  data, `share/locale`, `share/doc`, `man/man1`).
- `WITH_FHS=ON` on `main` configures, builds to completion, and a session
  started from `build/output/bin/grass` works (verified with
  `--tmp-project` and `g.version`). It works partly thanks to a
  configure-time symlink workaround that fakes a GISBASE-shaped runtime
  tree (`CMakeLists.txt` around line 186).
- Known gaps under `WITH_FHS=ON`, marked `TODO(FHS)` in the tree:
  - `gui/wxpython/CMakeLists.txt`: menudata/menustrings generation skipped
  - `man/CMakeLists.txt`: full doc index generation skipped
  - `locale/CMakeLists.txt`: `translation_status.json` install skipped
  - `general/CMakeLists.txt`: `ETCBINDIR`/`ETCDIR` split unfinished
- Defects found by smoke testing:
  - `GRASS_INSTALL_PYDIR` under FHS is the absolute `PYTHON_SITEARCH`, so
    the build stages Python under `build/output/<absolute path>` and
    `cmake --install` would write outside `CMAKE_INSTALL_PREFIX`.
  - `grass --config python_path` reports the legacy
    `libexec/grass/etc/python` under FHS instead of the real location.
- PR #5630 (nilason, open, mergeable) is the prep for code-level FHS: it
  introduces per-resource environment variables (`GRASS_ETCDIR`,
  `GRASS_LOCALEDIR`, `GRASS_FONTSDIR`, `GRASS_COLORSDIR`, `GRASS_GUIWXDIR`,
  `GRASS_MISCDIR`, ...), C accessors in `lib/gis/resource_dirs.c`
  (`G_etc_dir()` etc., fatal when unset), extends `RuntimePaths` and
  `resource_paths.py` (both build systems substitute values), and converts
  ~60 GISBASE-relative lookups across C and Python. Remaining
  `G_gisbase()` uses are for content that stays under the libexec GISBASE
  dir even in FHS (`driver/db`, `etc/lock`, `bin`, `scripts`), so they are
  correct as-is.
- PR #5630 blockers: Windows CI failure (`g.proj` exits with 0xC00000FD,
  suspected stack overflow, appeared after the October merge of `main`),
  one trivial unresolved review thread, re-review pending.
- Issue #5432 (configure writing outside the build dir with FHS) appears
  fixed on `main`; configure runs clean in a write-restricted sandbox.

## Phase 0 - land PR #5630

1. Debug the Windows `g.proj` crash (check whether it reproduces on `main`
   with the tests from #6482, or only on the PR branch).
2. Decide the env-var contract before merge: `resource_dirs.c` is fatal
   when a variable is unset. Alternative: fall back to compiled-in
   defaults so bare-GISBASE environments (embedded ctypes use, third-party
   wrappers) keep working. If fatal stays, call it out in release notes.
3. Resolve the last review thread, re-review, merge. State explicitly in
   the PR that addons install into `GRASS_ADDON_BASE`, which keeps the
   legacy user-dir layout regardless of the core layout.

Locally on this branch: the PR code is brought in as the Phase 0 commit.

## Phase 1 - make WITH_FHS=ON build complete (CMake only)

1. Make `GRASS_INSTALL_PYDIR` prefix-relative under FHS by default (e.g.
   `lib/pythonX.Y/site-packages` under the prefix), with an override
   option. Fixes staging under `build/output/` and `DESTDIR`/packaging.
2. Finish the `ETCDIR`/`ETCBINDIR` split (`general/CMakeLists.txt`):
   arch-independent data to `share/grass/etc`, executables to
   `libexec/grass/etc`; classify what each `etc/` install contains.
3. Remove the configure-time symlink workaround in `CMakeLists.txt`.
   After Phase 0, `grass_env_command` exports the `GRASS_*` variables, so
   build-time tool runs no longer need a GISBASE-shaped tree.
4. Re-enable the skipped targets under FHS: wxGUI menudata/menustrings,
   man full index, `translation_status.json` (install into
   `GRASS_INSTALL_MISCDIR`).
5. Audit FHS install destinations: headers, pkg-config, CMake package
   config, `build_addon.cmake` so `g.extension` works against an FHS
   install.

## Phase 2 - startup and runtime under an FHS install

1. Install to a prefix and verify `grass --config path` and
   `--config python_path` return layout-correct values in both modes.
2. Session smoke tests from the installed tree: `--tmp-project` plus a
   tool reading etc data (e.g. `r.colors`); this is the regression test
   that the env-var mechanism carries the load once symlinks are gone.
3. Decide the wxGUI layout question: under FHS the GUI lands in
   `site-packages/grass/gui` (importable package) rather than a GISBASE
   data tree; icons/images/xml must resolve via `GRASS_GUIRESDIR`. Needs a
   short design discussion upstream.
4. Verify libraries in the standard libdir are found without
   `LD_LIBRARY_PATH`; `runtime.py` dynamic-library-path setup should skip
   itself in that case.

## Phase 3 - validation and CI

1. Add a CI job: Ubuntu, `cmake -DWITH_FHS=ON`, build, install, run pytest
   and a gunittest subset against the installed FHS layout (not
   `build/output/`).
2. Keep legacy-CMake and Autotools jobs as the regression guard.
3. `DESTDIR` staged-install file-list check as a packaging smoke test.

## Phase 4 - docs and rollout

- Document `WITH_FHS` in `INSTALL.md`; note the env-var session contract
  in release notes; keep `OFF` as the default. FHS targets Linux
  packaging first; Windows/macOS stay on the legacy layout.

## Progress on this branch (2026-07-04)

- Phase 0: PR #5630 squashed in. Two fixes on top, both worth upstreaming
  separately:
  - Header staging in CMake had no dependency on source headers, so
    incremental builds compiled against stale headers (crashed via an
    implicitly declared, pointer-truncating G_locale_dir()).
  - G_init_locale() recursed without bound when GRASS_LOCALEDIR was
    unset (G_fatal_error translates, which re-enters locale init).
    Likely the Windows g.proj 0xC00000FD failure on the PR.
- Phase 1: done. WITH_FHS=ON and legacy both build completely; symlink
  workaround removed; GUI menudata, man index, and translation status
  enabled under FHS; GRASS_INSTALL_PYDIR prefix-relative; installed
  fontcap paths rewritten to the prefix; demolocation GISDBASE fixed.
- Phase 2: done. Added GRASS_PYDIR and GRASS_MANDIR resource paths (both
  build systems), used by session PYTHONPATH/MANPATH with legacy
  fallbacks; --config python_path reports the resolved path; g.manual
  converted to resource variables (missed by PR #5630). Verified with
  pytest (grass.app, grass.script, r.slope.aspect) against the FHS
  install and by session smoke tests on both layouts.
- Phase 3: done. CMake CI workflow now a WITH_FHS OFF/ON matrix.
- Phase 4: done. INSTALL.md CMake/WITH_FHS section; AGENTS.md
  LD_LIBRARY_PATH note for FHS.
- Still open (upstream design items): wxGUI-as-package decision,
  g.extension/addon build against an FHS install, pkg-config file for
  CMake builds (missing for both layouts), Windows/macOS layouts.

## Risks

- The env-var contract breaks third-party code that builds a GRASS
  environment by hand from `GISBASE` alone (failure is loud and
  self-explanatory).
- The GUI-as-package decision in Phase 2 is the only part not already
  settled by PR #5630.
