/*!
   \file lib/proj/db_info.c

   \brief GProj Library - Information about the PROJ database in use

   SPDX-FileCopyrightText: 2026 Vaclav Petras
   SPDX-FileCopyrightText: 2026 GRASS Development Team
   SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#endif

#include <grass/gis.h>
#include <grass/gprojects.h>
#include <grass/glocale.h>

#ifdef _WIN32
#define SEARCH_PATH_SEPARATOR ';'
#else
#define SEARCH_PATH_SEPARATOR ':'
#endif

// Return 1 if the file exists, 0 otherwise. PROJ uses paths in UTF-8, but
// on Windows, it may keep a path from an environment variable, such as
// PROJ_DATA, in the ANSI code page, so fall back to it when the path is not
// valid UTF-8.
static int file_exists(const char *path)
{
#ifdef _WIN32
    UINT code_page = CP_UTF8;
    int len =
        MultiByteToWideChar(code_page, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    wchar_t *wide_path;
    int exists;

    if (len <= 0) {
        code_page = CP_ACP;
        len = MultiByteToWideChar(code_page, 0, path, -1, NULL, 0);
    }
    if (len <= 0)
        return 0;
    wide_path = G_malloc(len * sizeof(wchar_t));
    MultiByteToWideChar(code_page, 0, path, -1, wide_path, len);
    exists = _waccess(wide_path, 0) == 0;
    G_free(wide_path);
    return exists;
#else
    return access(path, F_OK) == 0;
#endif
}

/*!
   \brief Find a proj.db file in the PROJ search path

   Looks at the directories PROJ searches for its resources, in the order
   PROJ uses them, and returns the first proj.db file found. This is useful
   when PROJ cannot open its database: if a proj.db file exists, PROJ
   rejected it, typically because it comes from another PROJ installation
   with an incompatible database layout.

   This calls proj_info(), which invalidates the search path string returned
   by any earlier proj_info() call.

   \return newly allocated path to be freed with G_free(), or NULL when
   there is no proj.db file in the search path
 */
char *GPJ_search_proj_db(void)
{
    char *searchpath = G_store(proj_info().searchpath);
    const char *start = searchpath;
    char *found = NULL;

    while (start && *start) {
        const char *end = strchr(start, SEARCH_PATH_SEPARATOR);
        size_t len = end ? (size_t)(end - start) : strlen(start);

        if (len > 0) {
            char *path = NULL;

            G_asprintf(&path, "%.*s%cproj.db", (int)len, start, HOST_DIRSEP);
            if (file_exists(path)) {
                found = path;
                break;
            }
            G_free(path);
        }
        if (!end)
            break;
        start = end + 1;
    }
    G_free(searchpath);
    return found;
}

// Return the name of the environment variable which points PROJ to its
// data: PROJ_DATA since PROJ 9.1, PROJ_LIB in PROJ 9.0. Remove this, and
// use PROJ_DATA directly, when GRASS requires PROJ 9.1 or later.
static const char *proj_data_variable(void)
{
    PJ_INFO info = proj_info();

    if (info.major < 9 || (info.major == 9 && info.minor < 1))
        return "PROJ_LIB";
    return "PROJ_DATA";
}

/*!
   \brief Report the state of the PROJ database (proj.db)

   Most PROJ calls fail when PROJ cannot open its database or opens one
   from another PROJ installation. Call this right before reporting a
   failed PROJ call.

   When a database is in use, its path is printed as an important message.
   Otherwise, a warning says whether a proj.db file was found but rejected,
   or which directories were searched, together with the value of the
   relevant environment variable and a hint how to fix it.
 */
void GPJ_report_db_status(void)
{
    PJ_LOG_LEVEL log_level;
    const char *db;
    char *db_path;
    char *found;
    char *searchpath;
    char version[64];
    const char *variable;
    const char *set_variable;
    const char *value;
    const char *user_dir;
    char *user_dir_copy;

    // The failed PROJ call of the caller has already logged the missing
    // database, so keep PROJ quiet here.
    log_level = proj_log_level(NULL, PJ_LOG_NONE);
    db = proj_context_get_database_path(NULL);
    // Valid only until the next PROJ call
    db_path = db ? G_store(db) : NULL;
    proj_log_level(NULL, log_level);

    if (db_path) {
        G_important_message(_("PROJ database in use: %s"), db_path);
        G_free(db_path);
        return;
    }

    variable = proj_data_variable();
    set_variable = variable;
    value = getenv(variable);
    // PROJ 9.1 and later still use PROJ_LIB when PROJ_DATA is not set.
    if (!value && strcmp(variable, "PROJ_LIB") != 0) {
        value = getenv("PROJ_LIB");
        if (value)
            set_variable = "PROJ_LIB";
    }
    found = GPJ_search_proj_db();
    user_dir = proj_context_get_user_writable_directory(NULL, 0);
    user_dir_copy = user_dir ? G_store(user_dir) : NULL;
    // Each proj_info() call frees the previous search path string,
    // so copy the values from the last call.
    {
        PJ_INFO info = proj_info();

        snprintf(version, sizeof(version), "%s", info.version);
        searchpath = G_store(info.searchpath);
    }
    if (found && user_dir_copy &&
        strncmp(found, user_dir_copy, strlen(user_dir_copy)) == 0 &&
        (found[strlen(user_dir_copy)] == HOST_DIRSEP ||
         found[strlen(user_dir_copy)] == '/')) {
        // PROJ searches the user directory first, so setting a variable
        // does not help.
        G_warning(_("PROJ %s cannot use the database <%s> in the PROJ user "
                    "directory, which probably comes from another PROJ "
                    "installation. Remove it or replace it with the proj.db "
                    "of PROJ %s."),
                  version, found, version);
        G_free(found);
    }
    else if (found) {
        G_warning(_("PROJ %s cannot use the database <%s>, which probably "
                    "comes from another PROJ installation. Set %s to the "
                    "directory containing the proj.db of PROJ %s."),
                  version, found, variable, version);
        G_free(found);
    }
    else if (value) {
        G_warning(_("PROJ cannot find its database (proj.db). Searched: %s "
                    "(%s is set to <%s>). Set %s to the directory containing "
                    "proj.db."),
                  searchpath, set_variable, value, variable);
    }
    else {
        G_warning(_("PROJ cannot find its database (proj.db). Searched: %s "
                    "(%s is not set). Set %s to the directory containing "
                    "proj.db."),
                  searchpath, variable, variable);
    }
    G_free(searchpath);
    G_free(user_dir_copy);
}
