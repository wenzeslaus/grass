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

/* The number of threads from the OMP_NUM_THREADS environment variable, or
 * 0 when it is not set to a positive number. The first value of a list
 * counts, with at most nine digits, as in resolve_nprocs() of
 * grass.script. OpenMP ignores a value which is not positive. */
static int omp_num_threads_env(void)
{
    const char *env = getenv("OMP_NUM_THREADS");
    int value = 0;
    int digits = 0;

    if (!env)
        return 0;
    while (*env == ' ' || *env == '\t')
        env++;
    for (; *env >= '0' && *env <= '9'; env++) {
        if (++digits > 9)
            return 0;
        value = 10 * value + (*env - '0');
    }
    while (*env == ' ' || *env == '\t')
        env++;
    return *env == '\0' || *env == ',' ? value : 0;
}

static int default_threads(void)
{
    static int threads = 0;

    if (threads < 1) {
#if defined(_OPENMP)
        threads = omp_num_threads_env() > 0 ? 0 : G_available_cores();
        if (threads < 1)
            threads = omp_get_max_threads();
#else
        threads = omp_num_threads_env();
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
 * A positive value is the number of threads. 0 means the default, which
 * is the number of physical cores available to the process (see
 * G_available_cores()), or the OpenMP default when the OMP_NUM_THREADS
 * environment variable is set to a positive number or the number of
 * cores is unknown. A value of OMP_NUM_THREADS which OpenMP ignores, such
 * as 0 or text, counts as not set. Without OpenMP, the default is the
 * first value of OMP_NUM_THREADS when that is a positive number, then the
 * number of physical cores, or 1 when that is unknown. The default is
 * determined once per process, on the first call. A negative value is
 * that many threads fewer than the default, at least one.
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
    const char *answer;
    long long value;
    char *end;

    if (nprocs == NULL)
        G_fatal_error(_("Option is NULL."));
    else if (nprocs->key == NULL)
        G_fatal_error(_("Option key is NULL."));
    answer = nprocs->answer ? nprocs->answer : "";
    errno = 0;
    value = strtoll(answer, &end, 10);
    if (errno || end == answer || *end != '\0' || value > INT_MAX ||
        value < INT_MIN)
        G_fatal_error(_("Invalid number of threads <%s> for %s="), answer,
                      nprocs->key);
    return G_resolve_nprocs((int)value);
}
