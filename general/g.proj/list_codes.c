#include <string.h>
#include <unistd.h>
#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/config.h>

#include <proj.h>

char *get_authority_names(void)
{
    char *authnames;
    int i, len;
    PROJ_STRING_LIST authlist;
    PJ_LOG_LEVEL log_level;

    /* This runs for every g.proj call while the options are defined.
     * Keep PROJ quiet here; operations which need the database repeat
     * the lookup and PROJ reports the reason then. PROJ reports the level
     * without its sign, so a negative PROJ_DEBUG level is restored as
     * positive, which only makes PROJ more verbose. */
    log_level = proj_log_level(NULL, PJ_LOG_NONE);
    authlist = proj_get_authorities_from_database(NULL);
    proj_log_level(NULL, log_level);

    if (!authlist) {
        /* PROJ cannot open its database. Leave the option values
         * unrestricted so that operations which do not need the list
         * still work. list_codes() reports the error when the list is
         * actually requested. */
        return NULL;
    }

    len = 0;
    for (i = 0; authlist[i]; i++) {
        len += strlen(authlist[i]) + 1;
    }
    if (len > 0) {
        authnames = G_malloc((len + 1) * sizeof(char)); /* \0 */
        *authnames = '\0';
        for (i = 0; authlist[i]; i++) {
            if (i > 0)
                strcat(authnames, ",");
            strcat(authnames, authlist[i]);
        }
    }
    else {
        authnames = G_store("");
    }
    proj_string_list_destroy(authlist);

    return authnames;
}

void list_codes(char *authname)
{
    int i, crs_cnt;
    PROJ_CRS_INFO **proj_crs_info;

    crs_cnt = 0;
    proj_crs_info =
        proj_get_crs_info_list_from_database(NULL, authname, NULL, &crs_cnt);
    if (crs_cnt < 1) {
        PJ_LOG_LEVEL log_level;
        const char *db_path;

        /* PROJ has already reported the missing database above. */
        log_level = proj_log_level(NULL, PJ_LOG_NONE);
        db_path = proj_context_get_database_path(NULL);
        proj_log_level(NULL, log_level);
        if (!db_path)
            G_fatal_error(
                _("Unable to list codes because PROJ cannot find its "
                  "database (proj.db). Check the PROJ installation or set "
                  "the PROJ_DATA environment variable (PROJ_LIB before "
                  "PROJ 9.1) to the directory containing proj.db."));
        G_fatal_error(_("No codes found for authority %s"), authname);
    }

    for (i = 0; i < crs_cnt; i++) {
        const char *proj_definition;
        char emptystr;
        PJ *pj;

        emptystr = '\0';
        pj = proj_create_from_database(NULL, proj_crs_info[i]->auth_name,
                                       proj_crs_info[i]->code, PJ_CATEGORY_CRS,
                                       0, NULL);
        proj_definition = proj_as_proj_string(NULL, pj, PJ_PROJ_5, NULL);
        if (!proj_definition) {
            /* what to do with a CRS without proj string ? */
            G_debug(1, "No proj string for %s:%s", proj_crs_info[i]->auth_name,
                    proj_crs_info[i]->code);
            proj_definition = &emptystr;
        }

        if (proj_definition) {
            fprintf(stdout, "%s|%s|%s\n", proj_crs_info[i]->code,
                    proj_crs_info[i]->name, proj_definition);
        }

        proj_destroy(pj);
    }
}
