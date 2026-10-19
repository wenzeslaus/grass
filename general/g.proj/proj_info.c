/*****************************************************************************
 *
 * MODULE:       g.proj
 * PURPOSE:      Report the PROJ installation in use: library version,
 *               database (proj.db) and the directories PROJ searches.
 * SPDX-FileCopyrightText: 2026 GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#include <stdio.h>

#include <grass/gis.h>
#include <grass/gprojects.h>
#include <grass/glocale.h>

#include "local_proto.h"

/* Strings returned by PROJ are valid only until the next call,
 * so each value is copied right away. */
static char *get_db_metadata(const char *key)
{
    const char *value = proj_context_get_database_metadata(NULL, key);

    return value ? G_store(value) : NULL;
}

static void print_plain(const char *label, const char *value)
{
    fprintf(stdout, "%-26s %s\n", label, value ? value : _("unknown"));
}

/* Print the PROJ version the database was created by and flag a different
 * major or minor version than the one in use. PROJ accepts databases with
 * a compatible layout, but such a mix usually means PROJ_DATA points to
 * the data of another PROJ installation. */
static void print_db_proj_version(const char *db_proj_version,
                                  int runtime_major, int runtime_minor)
{
    int major;
    int minor;
    char *text = NULL;

    if (db_proj_version &&
        sscanf(db_proj_version, "%d.%d", &major, &minor) == 2 &&
        (major != runtime_major || minor != runtime_minor)) {
        G_asprintf(&text, _("%s (differs from the PROJ version in use)"),
                   db_proj_version);
        print_plain(_("Database created by PROJ:"), text);
        G_free(text);
        return;
    }
    print_plain(_("Database created by PROJ:"), db_proj_version);
}

static void print_shell(const char *key, const char *value)
{
    fprintf(stdout, "%s=%s\n", key, value ? value : "");
}

/* Return the name of the environment variable which points PROJ to its
 * data: PROJ_DATA since PROJ 9.1, PROJ_LIB in PROJ 9.0. Remove this, and
 * use PROJ_DATA directly, when GRASS requires PROJ 9.1 or later. */
static const char *proj_data_variable(void)
{
    PJ_INFO info = proj_info();

    if (info.major < 9 || (info.major == 9 && info.minor < 1))
        return "PROJ_LIB";
    return "PROJ_DATA";
}

void print_proj_install_info(enum OutputFormat format)
{
    PJ_LOG_LEVEL log_level;
    char version[64];
    int runtime_major;
    int runtime_minor;
    char *searchpath;
    const char *db;
    char *db_path = NULL;
    char *db_rejected = NULL;
    char *db_layout = NULL;
    char *db_epsg = NULL;
    char *db_proj_version = NULL;
    const char *variable = proj_data_variable();

    /* Missing or unusable database is reported below, so keep PROJ's
     * own logger quiet. PROJ reports the level without its sign, so a
     * negative PROJ_DEBUG level is restored as positive, which only makes
     * PROJ more verbose. */
    log_level = proj_log_level(NULL, PJ_LOG_NONE);

    db = proj_context_get_database_path(NULL);
    if (db) {
        char *major;
        char *minor;

        db_path = G_store(db);
        major = get_db_metadata("DATABASE.LAYOUT.VERSION.MAJOR");
        minor = get_db_metadata("DATABASE.LAYOUT.VERSION.MINOR");
        if (major)
            G_asprintf(&db_layout, "%s.%s", major, minor ? minor : "0");
        G_free(major);
        G_free(minor);
        db_epsg = get_db_metadata("EPSG.VERSION");
        db_proj_version = get_db_metadata("PROJ.VERSION");
    }
    else {
        /* A proj.db in the search path which PROJ did not open */
        db_rejected = GPJ_search_proj_db();
    }
    proj_log_level(NULL, log_level);

    /* Each proj_info() call frees the previous search path string,
     * so take the values from one call after all others. */
    {
        PJ_INFO info = proj_info();

        snprintf(version, sizeof(version), "%s", info.version);
        runtime_major = info.major;
        runtime_minor = info.minor;
        searchpath = G_store(info.searchpath);
    }

    if (format == SHELL) {
        print_shell("proj_runtime", version);
        print_shell("proj_db", db_path);
        print_shell("proj_db_rejected", db_rejected);
        print_shell("proj_db_layout", db_layout);
        print_shell("proj_db_epsg", db_epsg);
        print_shell("proj_db_created_by_proj", db_proj_version);
        print_shell("proj_searchpath", searchpath);
    }
    else {
        print_plain(_("PROJ version:"), version);
        if (db_path) {
            print_plain(_("Database:"), db_path);
            print_plain(_("Database layout version:"), db_layout);
            print_plain(_("EPSG version:"), db_epsg);
            print_db_proj_version(db_proj_version, runtime_major,
                                  runtime_minor);
        }
        else if (db_rejected) {
            char *text = NULL;

            G_asprintf(&text,
                       _("%s cannot be used, it probably comes from another "
                         "PROJ installation"),
                       db_rejected);
            print_plain(_("Database:"), text);
            G_free(text);
        }
        else {
            char *text = NULL;

            G_asprintf(&text,
                       _("not found (set %s to the directory containing "
                         "proj.db)"),
                       variable);
            print_plain(_("Database:"), text);
            G_free(text);
        }
        print_plain(_("Search path:"), searchpath);
    }

    G_free(searchpath);
    G_free(db_path);
    G_free(db_rejected);
    G_free(db_layout);
    G_free(db_epsg);
    G_free(db_proj_version);
}
