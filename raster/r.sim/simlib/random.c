/* random.c (simlib), 20.nov.2002, JH */

#include <math.h>
#include <grass/gis.h>
#include <grass/simlib.h>

/*!
 * \brief Draw a pair of independent standard normal values
 *
 * Uses the polar method, which rejects points outside the unit circle, so
 * the number of values drawn from the stream varies, 8 / pi on average.
 */
void gasdev(struct G_random_state *stream, double *x, double *y)
{
    double r = 0.0, vv1 = 0.0, vv2 = 0.0, fac = 0.0;

    while (r >= 1. || r == 0.) {
        vv1 = G_random_double(stream) * 2. - 1.;
        vv2 = G_random_double(stream) * 2. - 1.;
        r = vv1 * vv1 + vv2 * vv2;
    }
    fac = sqrt(log(r) * -2. / r);
    (*y) = vv1 * fac;
    (*x) = vv2 * fac;
}
