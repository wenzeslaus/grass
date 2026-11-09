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

static int default_threads(void)
{
    static int threads = 0;

    if (threads < 1) {
#if defined(_OPENMP)
        const char *env = getenv("OMP_NUM_THREADS");

        threads = env && *env ? 0 : G_available_cores();
        if (threads < 1)
            threads = omp_get_max_threads();
#else
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
 * the number of physical cores available to the process (see
 * G_available_cores()), or the OpenMP default when the OMP_NUM_THREADS
 * environment variable is set or the number of cores is unknown. Without
 * OpenMP, the default is the number of physical cores, or 1 when that is
 * unknown. The default is determined once per process, on the first
 * call. A negative value is that many threads fewer than the default, at
 * least one.
 *
 * \param nprocs value of an nprocs option
 *
 * \return the number of threads, at least 1
 *
 * \since version 8.6
 */
int G_resolve_nprocs(int nprocs)
{
    /* Read for a positive value too, before its use in
     * omp_set_num_threads() changes what OpenMP reports. */
    int threads = default_threads();

    if (nprocs > 0)
        return nprocs;
    threads += nprocs;
    return threads < 1 ? 1 : threads;
}

/*!
 * \brief Read the number of threads from an nprocs option
 *
 * Call after G_parser(). The answer is resolved with G_resolve_nprocs().
 * An answer which is not an integer is a fatal error naming the option.
 * The function does not set up OpenMP threads, so the number can be
 * passed to any library.
 *
 * \param nprocs an nprocs option, usually G_OPT_M_NPROCS
 *
 * \return the number of threads, at least 1
 *
 * \since version 8.6
 */
int G_nprocs_from_option(const struct Option *nprocs)
{
    long long value;
    char *end;

    if (nprocs == NULL)
        G_fatal_error(_("Option is NULL."));
    else if (nprocs->key == NULL)
        G_fatal_error(_("Option key is NULL."));
    errno = 0;
    value = strtoll(nprocs->answer, &end, 10);
    if (errno || end == nprocs->answer || *end != '\0' || value > INT_MAX ||
        value < INT_MIN)
        G_fatal_error(_("Invalid number of threads <%s> for %s="),
                      nprocs->answer, nprocs->key);
    return G_resolve_nprocs((int)value);
}
