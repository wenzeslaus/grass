#include <stdio.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

/*! \brief Set the number of threads for OpenMP
    The intended usage is at the beginning of a C tool when parameters are
    processed, namely the G_OPT_M_NPROCS standard option.

    The number of threads is read from \em opt with G_nprocs_from_option().
    See G_resolve_nprocs() for the rules.

    \param opt A nprocs Option struct to specify the number of threads
    \return the number of threads set up for OpenMP parallel computing

    \since version 8.5
*/
int G_set_omp_num_threads(struct Option *opt)
{
    int threads = G_nprocs_from_option(opt);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
    G_verbose_message(n_("One thread is set up for parallel computing.",
                         "%d threads are set up for parallel computing.",
                         threads),
                      threads);
#else
    if (threads != 1) {
        G_warning(_("GRASS is not compiled with OpenMP support, parallel "
                    "computation is disabled. Only one thread will be used."));
    }
    threads = 1;
#endif
    return threads;
}
