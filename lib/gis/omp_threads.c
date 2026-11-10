#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

/*! \brief Set the number of threads for OpenMP
    The intended usage is at the beginning of a C tool when parameters are
    processed, namely the G_OPT_M_NPROCS standard option.

    If \em nprocs is 0, the default number of threads is used, which is
    the number of physical CPU cores available to the process (see
    G_available_cores()), or the OpenMP default when the OMP_NUM_THREADS
    environment variable is set or the number of cores is unknown. The
    default is determined once per process, on the first call.
    If \em nprocs is positive, that many threads are used.
    If \em nprocs is negative, it is added to the default (at least one
    thread is used), which keeps that many cores free for other work.

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
    /* Read once, before omp_set_num_threads() below changes what OpenMP
     * reports, so that every call in the process uses the same default. */
    static int base = 0;

    if (base < 1) {
        const char *env = getenv("OMP_NUM_THREADS");

        base = env && *env ? 0 : G_available_cores();
        if (base < 1)
            base = omp_get_max_threads();
    }
    if (threads < 1) {
        threads += base;
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
