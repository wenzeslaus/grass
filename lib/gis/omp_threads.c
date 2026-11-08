#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

#if defined(_OPENMP)
/* Whether the first value of OMP_NUM_THREADS is a positive number, the
 * same check as resolve_nprocs() in grass.script makes. OpenMP ignores
 * a zero or invalid value. */
static int omp_num_threads_is_set(void)
{
    const char *env = getenv("OMP_NUM_THREADS");
    char *end;
    long value;

    if (!env)
        return 0;
    while (isspace((unsigned char)*env))
        env++;
    if (!isdigit((unsigned char)*env))
        return 0;
    value = strtol(env, &end, 10);
    while (isspace((unsigned char)*end))
        end++;
    return value > 0 && (*end == '\0' || *end == ',');
}
#endif

/*! \brief Set the number of threads for OpenMP
    The intended usage is at the beginning of a C tool when parameters are
    processed, namely the G_OPT_M_NPROCS standard option.

    If \em nprocs is set to 0, the default number of threads is used: the
    value of the OMP_NUM_THREADS environment variable if it is set to a
    positive number, otherwise the number of physical CPU cores available
    to the process (see G_available_cores()).
    If \em nprocs is a positive number, specified number of threads is used.
    If \em nprocs is a negative number, then <em>default - number</em>
    threads are used, at least one (e.g. to keep \em number of cores free
    for other use).

    \param opt A nprocs Option struct to specify the number of threads
    \return the number of threads set up for OpenMP parallel computing

    \since version 8.5
*/
int G_set_omp_num_threads(struct Option *opt)
{
    /* make sure Option is not null */
    if (opt == NULL)
        G_fatal_error(_("Option is NULL."));
    else if (opt->key == NULL)
        G_fatal_error(_("Option key is NULL."));

    int threads = atoi(opt->answer);
#if defined(_OPENMP)
    if (threads < 1) {
        int default_threads = omp_num_threads_is_set() ? omp_get_max_threads()
                                                       : G_available_cores();

        threads += default_threads;
        threads = (threads < 1) ? 1 : threads;
    }
    omp_set_num_threads(threads);
    G_verbose_message(n_("One thread is set up for parallel computing.",
                         "%d threads are set up for parallel computing.",
                         threads),
                      threads);
#else
    if (!(threads == 0 || threads == 1)) {
        G_warning(_("GRASS is not compiled with OpenMP support, parallel "
                    "computation is disabled. Only one thread will be used."));
    }
    threads = 1;
#endif
    return threads;
}
