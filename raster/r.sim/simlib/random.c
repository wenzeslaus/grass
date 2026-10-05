/* random.c (simlib), 20.nov.2002, JH */

#include <inttypes.h>
#include <math.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include <grass/simlib.h>

/*!
 * \brief Return the seed of the walkers' random numbers
 *
 * The seed is read from the option or generated with the flag, and
 * reported in a verbose message; both together is a fatal error. Without
 * either, the seed is 12345, as it has been.
 *
 * \param seed the seed option
 * \param generate the flag to generate a seed
 *
 * \return the seed
 */
int64_t simwe_seed(const struct Option *seed, const struct Flag *generate)
{
    int64_t value;

    if (seed->answer && generate->answer)
        G_fatal_error(_("%s= and -%c are mutually exclusive"), seed->key,
                      generate->key);
    if (seed->answer) {
        value = G_random_seed_from_option(seed);
        G_verbose_message(_("Read random seed from %s option: %" PRId64),
                          seed->key, value);
    }
    else if (generate->answer) {
        value = G_random_generate_seed();
        G_verbose_message(_("Generated random seed (-s): %" PRId64), value);
    }
    else
        value = 12345;
    return value;
}

/*!
 * \brief Draw a pair of independent standard normal values
 *
 * Uses the polar method, which rejects points outside the unit circle, so
 * the number of values drawn varies, 8 / pi on average.
 *
 * \param state the random number state of the walker
 * \param[out] x the first value
 * \param[out] y the second value
 */
void gasdev(struct G_random_state *state, double *x, double *y)
{
    double r = 0.0, vv1 = 0.0, vv2 = 0.0, fac = 0.0;

    while (r >= 1. || r == 0.) {
        vv1 = G_random_double(state) * 2. - 1.;
        vv2 = G_random_double(state) * 2. - 1.;
        r = vv1 * vv1 + vv2 * vv2;
    }
    fac = sqrt(log(r) * -2. / r);
    (*y) = vv1 * fac;
    (*x) = vv2 * fac;
}
