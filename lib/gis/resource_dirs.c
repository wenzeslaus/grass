/*!
   \file lib/gis/resource_dirs.c

   \brief GIS Library - Get paths to resource directories.

   \author Nicklas Larsson

   (c) 2025 by the GRASS Development Team

   SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <grass/gis.h>
#include <grass/glocale.h>

static const char *get_g_env(const char *, const char *);

const char *G_colors_dir(void)
{
    return get_g_env("GRASS_COLORSDIR", "etc/colors");
}

const char *G_etcbin_dir(void)
{
    return get_g_env("GRASS_ETCBINDIR", "etc");
}

const char *G_etc_dir(void)
{
    return get_g_env("GRASS_ETCDIR", "etc");
}

const char *G_fonts_dir(void)
{
    return get_g_env("GRASS_FONTSDIR", "fonts");
}

const char *G_locale_dir(void)
{
    return get_g_env("GRASS_LOCALEDIR", "locale");
}

/*!
   \brief Get a resource directory from the environment.

   Returns the value of the environment variable when set. Otherwise,
   falls back to the given path relative to GISBASE, which is where the
   resource lives in the legacy (non-FHS) layout, so that environments
   created from GISBASE alone keep working with such installations.
   Fails fatally only when neither source yields a usable directory.
 */
static const char *get_g_env(const char *env_var, const char *gisbase_rel_dir)
{
    const char *value = getenv(env_var);
    if (value)
        return value;

    const char *gisbase = getenv("GISBASE");
    if (gisbase && *gisbase) {
        char path[GPATH_MAX];

        snprintf(path, sizeof(path), "%s/%s", gisbase, gisbase_rel_dir);
        if (access(path, F_OK) == 0)
            return G_store(path);
    }

    G_fatal_error(_("Incomplete GRASS session: Variable '%s' not set"),
                  env_var);
}
