# RFC 18 Bug Findings: g.extension Extraction

Status: Living document (findings from the RFC 18 stage 1 extraction)

Date: 2026-07-17

The stage 1 extraction of *g.extension* code into the `grass.addons`
library ([RFC 18](RFC18_addon_management.md)) surfaced a number of
pre-existing bugs. They were first preserved to keep the extraction
behavior-preserving and then fixed in a follow-up pass. This document
records each finding, its status, and where it is tested. Functions are
named instead of line numbers because the code moved.

## 1. Fixed

### 1.1 `file://` prefix stripped off by one character

- Where: `resolve_source_code` (now `grass.addons.resolve`).
- Symptom: `url[6:]` removed only 6 of the 7 characters of `file://`,
  so `file:///path` became `//path`. POSIX path handling tolerated the
  doubled slash, which is why it went unnoticed.
- Fix: `url.removeprefix("file://")`.
- Test: `test_resolve_source_code_file_url_prefix`.

### 1.2 Unreachable error handler in `get_default_branch`

- Where: `get_default_branch` (now `grass.addons.resolve`).
- Symptom: the organization/repository unpacking was guarded by
  `except URLError`, but unpacking raises `ValueError`, so the
  "Cannot retrieve organization and repository from URL" error was
  unreachable and a malformed URL crashed with a raw traceback.
- Fix: catch `ValueError` and raise `SourceResolutionError`. A URL
  without an organization/repository path (e.g. `https://github.com`)
  now produces the intended message.
- Test: `test_get_default_branch_without_organization_and_repository`.

### 1.3 `get_default_branch` crashed for unknown hosting services

- Where: `get_default_branch` (now `grass.addons.resolve`).
- Symptom: the docstring promises "main" as the default for unknown
  services, but an unknown host reached `urlopen(None)` and crashed
  with a non-URLError exception.
- Fix: look up the API URL first and return "main" immediately when
  the host is not a known service (no network access in that case).
- Test: `test_get_default_branch_unknown_host`.

### 1.4 Implicit `None` return for unsupported local files

- Where: `resolve_source_code` (now `grass.addons.resolve`).
- Symptom: an existing local file matching neither `.zip` nor a
  supported tar suffix fell through both branches and returned `None`,
  crashing the caller on tuple unpacking, despite the docstring
  promising a tuple.
- Fix: raise `SourceResolutionError` naming the file and the supported
  archive formats.
- Test: `test_resolve_source_code_local_unsupported_archive`.

### 1.5 Correlate codes lost when writing toolboxes metadata

- Where: `write_xml_toolboxes` (now
  `grass.addons.registry.LocalRegistry`).
- Symptom: the correlate line wrote `tnode.get("code")` (the
  toolbox's own code) instead of `cnode.get("code")`, so every
  `<correlate>` entry was overwritten with the parent toolbox code on
  each write of `toolboxes.xml`.
- Fix: write the correlate node's own code.
- Test: correlate round-trip test; the golden file
  `golden_toolboxes.xml` pins the corrected output (the only deltas
  against the pre-fix golden are the two correlate lines).

### 1.6 Literal string "None" written into modules metadata

- Where: `write_xml_modules` (now
  `grass.addons.registry.LocalRegistry`).
- Symptom: empty `<description>` or `<keywords>` elements round-tripped
  as the literal text `None` in the written file (the fixture-derived
  golden contained 14 such occurrences), polluting data consumed by
  *g.search.modules* and listings.
- Fix: render an empty string when the element text is `None`.
- Test: empty-optional-elements test; `golden_modules.xml` pins the
  corrected output.

### 1.7 Install prefix re-prepended on every metadata write

- Where: `write_xml_modules` and `write_xml_extensions` (now
  `grass.addons.registry.LocalRegistry`).
- Symptom: file entries were unconditionally passed through
  `os.path.join(prefix, entry)` on every write. This was idempotent
  only because real prefixes are absolute (join discards the first
  argument for an absolute second argument); with a relative prefix,
  every rewrite prepended the prefix again.
- Fix: join only entries which are not absolute and not already under
  the prefix.
- Test: rewrite-idempotence tests parametrized over absolute and
  relative prefixes.

### 1.8 Missing toolboxes file created with the modules DOCTYPE

- Where: `install_toolbox_xml` (now
  `grass.addons.registry.LocalRegistry`).
- Symptom: a missing `toolboxes.xml` was created with the modules
  writer, so the transient empty file carried
  `<!DOCTYPE task ...>` instead of `<!DOCTYPE toolbox ...>`. The final
  file was written correctly, so the effect was internal only.
- Fix: create the file with the toolboxes writer.
- Test: DOCTYPE test on the created file.

### 1.9 Listing installed toolboxes wrote to the prefix

- Where: `get_installed_toolboxes` (now
  `grass.addons.registry.LocalRegistry`).
- Symptom: unlike `get_installed_modules`, which honors its *force*
  parameter, the toolboxes reader always created a missing
  `toolboxes.xml` — a listing operation should not write.
- Fix: mirror the modules semantics (create only with `force=True`,
  otherwise report and return an empty list).
- Test: split missing-file tests with and without force.

### 1.10 Locale-dependent encoding despite declared UTF-8

- Where: all readers and writers of the three metadata files (now
  `grass.addons.registry`).
- Symptom: files declaring `encoding="UTF-8"` in their XML prolog were
  read and written with the locale default encoding, corrupting
  non-ASCII descriptions on non-UTF-8 locales.
- Fix: pass `encoding="utf-8"` explicitly everywhere.
- Test: UTF-8 round-trip test with non-ASCII text.

### 1.11 Cosmetic code fixes

- `resolve_install_prefix` appended a trailing separator "for URL
  pasting" and then immediately stripped it via `os.path.abspath` — a
  no-op with a misleading comment; removed (now
  `grass.addons.config`).
- Redundant `IOError` alias in exception clauses (`IOError` is
  `OSError`); removed.
- Garbled docstring wording in `resolve_known_host_service`; fixed.
- `install_toolbox_xml` in the tool was dead code (its only call site
  had been commented out); the tool-side copy was removed, the library
  method remains for future toolbox work.

## 2. Intentional behavior changes (not bug fixes)

### 2.1 SVN support removed

Requested decision: legacy Subversion support is dropped in favor of
work on server switching and packaging.

- `resolve_source_code` no longer falls back to the `svn` source type
  for unrecognized URLs. It returns the new plain-`url` source type
  instead: the URL remains usable as a metadata location, so *listing
  addons from an alternative server URL keeps working* (this was the
  useful behavior hiding behind the old svn fallback). Installing from
  a plain URL fails with a clear message naming the supported sources
  (archives, known hosting services, local paths) instead of
  attempting `svn checkout`.
- The dead SVN HTML-scraper listing fallbacks
  (`list_available_extensions_svn`, `get_wxgui_extensions`) and the
  `download_source_code_svn` function were deleted from *g.extension*.
  A failure to fetch or parse a server's `modules.xml` is now a fatal
  error instead of a silent fallback to HTML scraping.
- The option description no longer claims "If not identified,
  Subversion repository is assumed."
- The live-network Subversion doctest disappeared with the fallback.

### 2.2 Sharper errors

Malformed repository URLs and unsupported local files now produce
clear `SourceResolutionError`-based fatal messages where they
previously crashed with tracebacks or unhelpful fallbacks (see 1.2,
1.4).

## 3. Known issues, deferred

- **Toolbox bookkeeping is inconsistent**: `remove_extension_xml` uses
  `len(edict) > 1` as an "is a toolbox" heuristic and passes the
  extension *name* where toolbox *codes* are matched. Toolbox support
  is documented as experimental; a proper fix needs the toolbox design
  work planned for a later RFC 18 stage.
- **Reinstalls never refresh metadata**: `install_extension_xml` and
  `install_module_xml` skip existing entries with only a verbose
  "metadata not updated!" note, so reinstalling an addon does not
  update its recorded description, keywords, or libgis revision except
  through prior removal.
- **Listing requires a writable prefix**: `resolve_install_prefix` runs
  (and checks writability) even for pure listing operations, so
  `g.extension -l` fails when the addon directory is not writable.
  Addressed by the sessionless/read-only operations design in the
  RFC 18 plan (stages 3-4).
- **Nonexistent local paths are treated as remote URLs**: inherent
  ambiguity documented in the `resolve_source_code` docstring; a typo
  in a local path produces a remote-URL error.
- **Stale doctest examples**: the `# doctest: +SKIP` examples in
  `resolve_source_code` still show 2015-era `master.zip`/`default.zip`
  outputs which predate default-branch discovery.
- **Repeated `gs.version()` subprocess calls**: the tool constructs
  registry access per operation and the version lookup runs a
  subprocess each time; consolidation is part of the state-hygiene
  stage of the RFC 18 plan.
