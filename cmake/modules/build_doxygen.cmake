#[[
AUTHOR(S):  GRASS Development Team
PURPOSE:    Add an opt-in target that builds the Doxygen programmer's manual.
            Enabled by the WITH_DOXYGEN option; this file is included from the
            top-level CMakeLists.txt only when that option is ON. The target is
            not part of ALL and is built explicitly with:
                cmake --build <build> --target doxygen
COPYRIGHT:  (C) 2026 by the GRASS Development Team

SPDX-License-Identifier: GPL-2.0-or-later
#]]

# Graphviz dot is optional: found means class and call graphs are drawn,
# missing means the manual is still built without them.
find_package(Doxygen OPTIONAL_COMPONENTS dot)

if(NOT DOXYGEN_FOUND)
    # WITH_DOXYGEN only adds a developer convenience target and gates no build
    # artifact, so a missing tool warns and skips rather than failing configure
    # the way a missing library dependency does.
    message(
        WARNING
        "WITH_DOXYGEN is ON but Doxygen was not found; the 'doxygen' target will not be available. Install Doxygen (and Graphviz/dot for graphs) to build the programmer's manual."
    )
    return()
endif()

if(TARGET Doxygen::dot)
    set(DOXYGEN_HAVE_DOT YES)
else()
    set(DOXYGEN_HAVE_DOT NO)
    message(
        STATUS
        "Doxygen found without Graphviz/dot; the programmer's manual will be built without graphs."
    )
endif()

set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doxygen)
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/Doxyfile.in
    ${DOXYGEN_OUTPUT_DIR}/Doxyfile
    @ONLY
)

add_custom_target(
    doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUTPUT_DIR}/Doxyfile
    WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIR}
    COMMENT "Building the Doxygen programmer's manual (HTML)"
    VERBATIM
)
set_target_properties(doxygen PROPERTIES FOLDER Docs)
