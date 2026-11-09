/*!
 * \file lib/gis/nprocs_option.c
 *
 * \brief GIS Library - Number of threads from the nprocs option of a tool
 *
 * SPDX-FileCopyrightText: 2026 Vaclav Petras
 * SPDX-FileCopyrightText: GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

/* The default number of threads: the OpenMP default when OMP_NUM_THREADS
 * is set or the cores are unknown, otherwise the number of physical cores.
 * Without OpenMP, the first value of OMP_NUM_THREADS is used when it is
 * positive. Read once, so that a second call in the same process adds to
 * the same default and not to the threads set by the first. */
static int default_threads(void)
{
    static int threads = 0;

    if (threads < 1) {
        const char *env = getenv("OMP_NUM_THREADS");

#if defined(_OPENMP)
        threads = env && *env ? 0 : G_available_cores();
        if (threads < 1)
            threads = omp_get_max_threads();
#else
        threads = env && *env ? atoi(env) : 0;
        if (threads < 1)
            threads = G_available_cores();
        if (threads < 1)
            threads = 1;
#endif
    }
    return threads;
}

/*!
 * \brief Resolve a value of nprocs into a number of threads
 *
 * A positive value is the number of threads. 0 means the default number:
 * the OpenMP default when the OMP_NUM_THREADS environment variable is
 * set, otherwise the number of physical cores available to the process
 * (see G_available_cores()). A negative value is that many threads fewer
 * than the default, at least one.
 *
 * \param nprocs value of an nprocs option
 *
 * \return the number of threads, at least 1
 */
int G_resolve_nprocs(int nprocs)
{
    int threads;

    if (nprocs > 0)
        return nprocs;
    threads = default_threads() + nprocs;
    return threads < 1 ? 1 : threads;
}

/*!
 * \brief Read the number of threads from an nprocs option
 *
 * Call after G_parser(). The answer is resolved with G_resolve_nprocs().
 * An answer which is not an integer is a fatal error naming the option.
 *
 * Unlike G_set_omp_num_threads(), this does not depend on OpenMP and does
 * not set up any threads, so tools can use it for the threads of other
 * libraries, such as GDAL.
 *
 * \param nprocs the nprocs option, usually G_OPT_M_NPROCS, whatever its key
 *
 * \return the number of threads, at least 1
 */
int G_nprocs_from_option(const struct Option *nprocs)
{
    long long value;
    char *end;

    if (!nprocs->answer)
        G_fatal_error(_("%s= is required"), nprocs->key);
    errno = 0;
    value = strtoll(nprocs->answer, &end, 10);
    if (errno || end == nprocs->answer || *end != '\0' || value > INT_MAX ||
        value < INT_MIN)
        G_fatal_error(_("Invalid number of threads <%s> for %s="),
                      nprocs->answer, nprocs->key);
    return G_resolve_nprocs((int)value);
}
