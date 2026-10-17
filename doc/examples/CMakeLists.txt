# Build the example modules under doc/examples as a compile check. This is
# reached only when WITH_EXAMPLES is ON (see the top-level CMakeLists.txt); the
# examples never build by default.
#
# Every immediate subdirectory of a group directory (raster/, vector/) that
# contains a main.c is built as one example module, so examples added later -
# such as proposed segment-library examples (r.example.segment,
# r.example.segmulti) under doc/examples/raster/ - are picked up without editing
# this file. Link libraries are declared per group rather than per module: each
# list is the union of the GRASS libraries used by that group's examples (raster
# examples also link grass_segment for the segment examples above). This keeps
# auto-discovery self-contained; a new example group would need its own entry
# here.
#
# NO_DOCS keeps this an opt-in compile check: the modules are compiled, linked,
# and staged like any tool, but their manual pages are not generated and they
# are not added to the documentation index.

set(raster_example_DEPENDS grass_gis grass_raster grass_segment)
set(vector_example_DEPENDS
    grass_dbmibase
    grass_dbmiclient
    grass_gis
    grass_vector
)

set(example_targets)
foreach(group raster vector)
    file(
        GLOB group_dirs
        LIST_DIRECTORIES TRUE
        "${CMAKE_CURRENT_SOURCE_DIR}/${group}/*"
    )
    foreach(example_dir ${group_dirs})
        if(EXISTS "${example_dir}/main.c")
            get_filename_component(example_name "${example_dir}" NAME)
            build_program_in_subdir(
                ${group}/${example_name}
                DEPENDS ${${group}_example_DEPENDS}
                NO_DOCS
            )
            set_target_properties(${example_name} PROPERTIES FOLDER Examples)
            list(APPEND example_targets ${example_name})
        endif()
    endforeach()
endforeach()

if(NOT example_targets)
    message(WARNING "WITH_EXAMPLES is ON but no example modules were found")
endif()

add_custom_target(
    ALL_EXAMPLE_MODULES
    DEPENDS ${example_targets}
    COMMENT "Building example modules under doc/examples."
)
set_target_properties(ALL_EXAMPLE_MODULES PROPERTIES FOLDER Examples)
