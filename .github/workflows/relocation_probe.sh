#!/usr/bin/env bash
# Probe what works in an installation moved away from its configured prefix.
# Usage: relocation_probe.sh <prefix> [inplace]
# With "inplace", the installation is not moved (control run).
set -u

prefix="$(realpath "$1")"
mode="${2:-moved}"
work="$(mktemp -d)"
mkdir "${work}/config"

launcher="${prefix}/bin/grass"
clean=(env -i "PATH=${PATH}" "HOME=${work}" "GRASS_CONFIG_DIR=${work}/config" "LANG=C.UTF-8")

# Layout-specific locations are taken before the move.
gisbase="$("${clean[@]}" "${launcher}" --config path)"
python_path="$("${clean[@]}" "${launcher}" --config python_path)"
case "${gisbase}" in
"${prefix}"*) ;;
*)
    echo "FAIL setup: path ${gisbase} is not under ${prefix}"
    exit 0
    ;;
esac

# The project is created before the move, so that it does not depend on it.
project="${work}/project"
"${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import sys
import grass.script as gs
gs.create_project(sys.argv[1])
" "${project}" || echo "FAIL setup: create_project before the move"

if [ "${mode}" = "moved" ]; then
    moved="${prefix}-moved"
    if [ -e "${moved}" ]; then
        echo "FAIL setup: ${moved} exists"
        exit 0
    fi
    mv "${prefix}" "${moved}"
    trap 'mv "${moved}" "${prefix}"; rm -rf -- "${work}"' EXIT
else
    moved="${prefix}"
    trap 'rm -rf -- "${work}"' EXIT
fi
launcher="${moved}/bin/grass"
gisbase="${moved}${gisbase#"${prefix}"}"
python_path="${moved}${python_path#"${prefix}"}"

echo "prefix: ${prefix} (${mode})"
echo "gisbase: ${gisbase}"
echo "python_path: ${python_path}"

# probe <name> <expected substring or empty> <command...>
probe() {
    local name="$1" expected="$2" output status
    shift 2
    output="$("$@" 2>&1)"
    status=$?
    if [ "${status}" -eq 0 ] && { [ -z "${expected}" ] || [[ "${output}" == *"${expected}"* ]]; }; then
        echo "PASS ${name}"
    else
        echo "FAIL ${name} (exit ${status}): $(echo "${output}" | grep -v '^$' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
    fi
}

# Launcher of the moved installation, nothing set.
probe "launcher: --config path" "${moved}" "${clean[@]}" "${launcher}" --config path
probe "launcher: --config python_path" "${moved}" "${clean[@]}" "${launcher}" --config python_path
probe "launcher: C tool in session" "projection" "${clean[@]}" "${launcher}" --tmp-project XY --exec g.region -p
probe "launcher: C tool using etc (r.colors -l)" "viridis" "${clean[@]}" "${launcher}" --tmp-project XY --exec r.colors -l
probe "launcher: Python tool (r.fillnulls)" "<task name=\"r.fillnulls\">" "${clean[@]}" "${launcher}" --tmp-project XY --exec r.fillnulls --interface-description
probe "launcher: grass.lib in session" "PERMANENT" "${clean[@]}" "${launcher}" --tmp-project XY --exec python3 -c "
import grass.lib.gis as libgis
libgis.G_gisinit('probe')
print(libgis.G_mapset())
"

# Python API with only PYTHONPATH pointing to the moved package.
probe "python: init, C tool" "projection" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import sys
import grass.script as gs
with gs.setup.init(sys.argv[1]) as session:
    print(gs.read_command('g.region', flags='p', env=session.env))
" "${project}"
probe "python: init, C tool using etc" "viridis" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import sys
import grass.script as gs
with gs.setup.init(sys.argv[1]) as session:
    print(gs.read_command('r.colors', flags='l', env=session.env))
" "${project}"
probe "python: init, grass.lib" "PERMANENT" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import sys
import grass.script as gs
import grass.lib.gis as libgis
with gs.setup.init(sys.argv[1]):
    libgis.G_gisinit('probe')
    print(libgis.G_mapset())
" "${project}"
probe "python: init, GISBASE in session env" "${moved}" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import sys
import grass.script as gs
with gs.setup.init(sys.argv[1]) as session:
    print(session.env['GISBASE'])
" "${project}"
probe "python: init, GRASS_ETCDIR in session env" "${moved}" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import sys
import grass.script as gs
with gs.setup.init(sys.argv[1]) as session:
    print(session.env.get('GRASS_ETCDIR'))
" "${project}"
probe "python: grass.lib import without session" "" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import grass.lib.gis
import grass.lib.raster
"
probe "python: grass.lib etc dir without session" "${moved}" "${clean[@]}" "PYTHONPATH=${python_path}" python3 -c "
import grass.lib.gis as libgis
print(libgis.G_etc_dir())
"

# Tools started directly, with only the session file given.
gisrc="${work}/gisrc"
printf 'GISDBASE: %s\nLOCATION_NAME: project\nMAPSET: PERMANENT\n' "${work}" >"${gisrc}"
probe "direct: C tool with GISRC only" "PERMANENT" "${clean[@]}" "GISRC=${gisrc}" "${gisbase}/bin/g.gisenv" get=MAPSET
probe "direct: C tool using etc with GISRC only" "viridis" "${clean[@]}" "GISRC=${gisrc}" "${gisbase}/bin/r.colors" -l
