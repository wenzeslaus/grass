/*!
   \file lib/proj/db_info.c

   \brief GProj Library - Information about the PROJ database in use

   (C) 2026 by the GRASS Development Team

   This program is free software under the GNU General Public License
   (>=v2). Read the file COPYING that comes with GRASS for details.
 */

#include <stdlib.h>

#include <grass/gis.h>
#include <grass/gprojects.h>
#include <grass/glocale.h>

/*!
   \brief Describe the state of the PROJ database (proj.db) for error messages

   PROJ needs its database (proj.db) for almost everything, and the most
   common reason for PROJ calls failing is that it cannot find the database,
   or finds one from a different PROJ version, e.g., when the PROJ_DATA
   environment variable points elsewhere or is missing in a relocated
   installation. PROJ itself reports the path only in some cases, so this
   function produces a one-line description to append to error and warning
   messages: the path of the database in use when there is one, otherwise
   the directories PROJ searched, the value of PROJ_DATA, and a hint how
   to fix it.

   \return newly allocated string, to be freed by the caller with G_free()
 */
char *GPJ_proj_db_status(void)
{
    const char *db_path = proj_context_get_database_path(NULL);
    char *status = NULL;

    if (db_path) {
        G_asprintf(&status, _("PROJ database in use: %s"), db_path);
    }
    else {
        PJ_INFO info = proj_info();
        const char *env_name = "PROJ_DATA";
        const char *env_value = getenv(env_name);

        if (!env_value) {
            env_name = "PROJ_LIB";
            env_value = getenv(env_name);
        }
        if (env_value) {
            G_asprintf(&status,
                       _("PROJ cannot open its database (proj.db). "
                         "Searched: %s (%s is set to '%s'). "
                         "Set PROJ_DATA to the directory containing proj.db."),
                       info.searchpath, env_name, env_value);
        }
        else {
            G_asprintf(&status,
                       _("PROJ cannot open its database (proj.db). "
                         "Searched: %s (PROJ_DATA is not set). "
                         "Set PROJ_DATA to the directory containing proj.db."),
                       info.searchpath);
        }
    }

    return status;
}
