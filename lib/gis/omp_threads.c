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

    If \em nprocs is set to 0, the default number of threads is used: the
    value of the OMP_NUM_THREADS environment variable if it is set,
    otherwise the number of physical CPU cores available to the process
    (see G_available_cores()). Threads on logical CPUs of the same physical
    core (SMT, Hyper-Threading) share its execution units, so threads
    beyond the number of physical cores add little speed for typical
    raster computations and slow down the other threads.
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
        const char *env = getenv("OMP_NUM_THREADS");
        /* OpenMP takes the default number of threads from OMP_NUM_THREADS
         * when it is set. */
        int default_threads =
            (env && *env) ? omp_get_max_threads() : G_available_cores();

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
