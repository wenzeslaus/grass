/*!
 * \file lib/gis/random_options.c
 *
 * \brief GIS Library - Random number seed from the options of a tool
 *
 * SPDX-FileCopyrightText: 2026 GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <errno.h>
#include <stdlib.h>

#include <grass/gis.h>
#include <grass/glocale.h>

/*!
 * \brief Get the random number seed from a seed option or a flag
 *
 * Call after G_parser(). Exactly one of the option and the flag must be
 * given; both or neither is a fatal error naming the two. The tool may
 * also declare them with G_option_exclusive() and G_option_required(), but
 * the check here makes every tool which uses the function behave the same.
 * When the tool has no flag to generate a seed, pass NULL, and the option
 * is then required.
 *
 * With the option, its answer is read with strtoll(). An answer which is
 * not an integer, has anything after the integer, or is outside the range
 * from -2^31 to 2^32 - 1 is a fatal error naming the option. Leading white
 * space and a sign are allowed. With the flag, the seed is what
 * G_random_generate_seed() returns, so the environment variables
 * GRASS_RANDOM_SEED and SOURCE_DATE_EPOCH apply.
 *
 * Either way, the result is a seed G_random_state_from_seed(), the layout
 * functions and G_srand48() accept. The function neither prints the seed
 * nor records it; the tool records it, for example in the history of its
 * output, so that the computation can be repeated.
 *
 * \param seed the seed option, usually G_OPT_M_SEED, whatever its key
 * \param generate the flag which asks for a generated seed, or NULL
 *
 * \return the seed, from -2^31 to 2^32 - 1
 */
long long G_random_seed_from_options(const struct Option *seed,
                                     const struct Flag *generate)
{
    long long value;
    char *end;

    if (generate) {
        if (seed->answer && generate->answer)
            G_fatal_error(_("%s= and -%c are mutually exclusive"), seed->key,
                          generate->key);
        if (!seed->answer && !generate->answer)
            G_fatal_error(_("Either %s= or -%c is required"), seed->key,
                          generate->key);
        if (generate->answer)
            return G_random_generate_seed();
    }
    else if (!seed->answer)
        G_fatal_error(_("%s= is required"), seed->key);

    errno = 0;
    value = strtoll(seed->answer, &end, 10);
    if (end == seed->answer || *end != '\0')
        G_fatal_error(_("Invalid random seed <%s> for %s=: not an integer"),
                      seed->answer, seed->key);
    if (errno == ERANGE || value < -(1LL << 31) || value > (1LL << 32) - 1)
        G_fatal_error(_("Invalid random seed <%s> for %s=: outside the range "
                        "from -2147483648 to 4294967295"),
                      seed->answer, seed->key);
    return value;
}
