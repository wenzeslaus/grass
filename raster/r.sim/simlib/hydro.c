/****************************************************************************
 *
 * MODULE:       simwe library
 * AUTHOR(S):    Helena Mitasova, Jaro Hofierka, Lubos Mitas:
 * PURPOSE:      Hydrologic and sediment transport simulation (SIMWE)
 *
 * SPDX-FileCopyrightText: 2002 GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

/* hydro.c (simlib), 20.nov.2002, JH */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <grass/gis.h>
#include <grass/bitmap.h>
#include <grass/linkm.h>
#include <grass/glocale.h>

#include <grass/simlib.h>
/*
 * Soeren 8. Mar 2011 TODO:
 * Put all these global variables into several meaningful structures and
 * document use and purpose.
 *
 */

struct point2D;
struct point3D;

/* **************************************************** */
/*       create walker representation of si */
/* ******************************************************** */
/*                       .......... iblock loop */

void main_loop(const Setup *setup, const Geometry *geometry,
               const Settings *settings, Simulation *sim,
               ObservationPoints *points, const Inputs *inputs,
               const Outputs *outputs, Grids *grids)
{
    int i, l, k;
    int iblock;
    double conn = 1.0;
    double addac;

    // nblock is reserved for Monte Carlo replicas. A future
    // change will allow nblock > 1, give each replica an
    // rwalk-sized walker subset and potentially run replicas concurrently and
    // reduce into the shared grids after the iblock loop. The factor and
    // conn formulas below already encode that design (factor's denominator
    // (rwalk * nblock) equals total walkers; conn = nblock/iblock scales a
    // sequential cumulative partial sum to an estimator of the eventual
    // total). With nblock = 1 today, the loop is a no-op wrapper and conn
    // collapses to 1.0. The historical auto-split based on a static MAXW
    // cap was removed: it broke per-block walker accounting (rwalk was not
    // actually divided), produced biased depth/discharge for users.
    int nblock = 1;

    double stxm = geometry->stepx * (double)(geometry->mx + 1) - geometry->xmin;
    double stym = geometry->stepy * (double)(geometry->my + 1) - geometry->ymin;
    double deldif = sqrt(setup->deltap) * settings->frac; /* diffuse factor */

    double factor =
        setup->deltap * setup->sisum / (sim->rwalk * (double)nblock);

    G_debug(2, " deldif, factor %f %e", deldif, factor);
    G_debug(2, " maxwa, nblock %d %d", sim->maxwa, nblock);
    G_debug(2, "rwalk, sisum: %f %f", sim->rwalk, setup->sisum);

    // Each walker draws from a stream of its own for the whole simulation,
    // so its path does not depend on which thread moves it. The layout has
    // a stream for every allocated walker, a number fixed by the region and
    // the nwalkers option. With more than one block, each block would need
    // streams of its own, numbered over all blocks.
    struct G_random_state stream_check;
    long long stream_length = G_random_seed_stream(
        &stream_check, settings->seed, 0, sim->max_walkers);

    // A walker draws two values when placed and in each time step a normal
    // pair by rejection, 8 / pi values on average, and one more in a trap.
    // Eight values per step is over twice the average, a total a walker is
    // practically certain not to reach over many steps; over few steps the
    // streams are far longer than needed.
    long long max_draws = 2 + 8LL * setup->miter;

    // A stream drawn past the length meets a lagged twin: a stream a
    // quarter, a half or three quarters of the walkers further, whose
    // values are this stream's plus a constant after a lag. Walkers are
    // numbered row by row, so the walkers of a twin pair start at least
    // about a quarter of the walkers apart, in different parts of the
    // region. The shifted values give other normal deviates, and the
    // polar method's rejection moves them into other pairs and steps, so
    // the paths are not copies of each other. That is why the check only
    // warns.
    if (stream_length < max_draws)
        G_warning(_("The random number streams of %d walkers hold %lld "
                    "values each, fewer than the %lld a walker may draw "
                    "in %d time steps, so the random numbers of "
                    "different walkers may overlap. Use fewer walkers "
                    "or a shorter simulation."),
                  sim->max_walkers, stream_length, max_draws, setup->miter);

    for (iblock = 1; iblock <= nblock; iblock++) {
        int lw = 0;
        double walkwe = 0.;

        G_message(_("Processing block %d of %d"), iblock, nblock);

        /* write hh.walkers0 */

        for (k = 0; k < geometry->my; k++) {
            for (l = 0; l < geometry->mx; l++) { /* run thru the whole area */
                if (grids->zz[k][l] != UNDEF) {

                    double x = geometry->xp0 + geometry->stepx * (double)(l);
                    double y = geometry->yp0 + geometry->stepy * (double)(k);

                    double gen = sim->rwalk * grids->si[k][l] / setup->sisum;
                    int mgen = (int)gen;
                    double wei = gen / (double)(mgen + 1);

                    for (int iw = 1; iw <= mgen + 1;
                         iw++) { /* assign walkers */
                        struct G_random_state *stream = &sim->streams[lw];

                        G_random_seed_stream(stream, settings->seed, lw,
                                             sim->max_walkers);
                        sim->w[lw].x = x + geometry->stepx *
                                               (G_random_double(stream) - 0.5);
                        sim->w[lw].y = y + geometry->stepy *
                                               (G_random_double(stream) - 0.5);
                        sim->w[lw].m = wei;

                        walkwe += sim->w[lw].m;
                        sim->vavg[lw].x = grids->v1[k][l];
                        sim->vavg[lw].y = grids->v2[k][l];
                        lw++;
                    }
                } /* defined area */
            }
        }
        sim->nwalk = lw;
        G_debug(2, " nwalk %d", sim->nwalk);
        G_debug(2, " walkwe (walk weight),frac %f %f", walkwe, settings->frac);

        sim->nwalka = 0;
        int nwalka = 0;

        // conn scales the cumulative partial sum in gama into an estimator
        // of the eventual total when blocks run sequentially.
        conn = (double)nblock / (double)iblock;

        /* ********************************************************** */
        /*       main loop over the projection time */
        /* *********************************************************** */

        G_debug(2, "main loop over the projection time... ");

        for (i = 1; i <= setup->miter;
             i++) { /* iteration loop depending on simulation time and deltap */
            G_percent(i, setup->miter, 1);
            if (setup->iterout > 0 && i % setup->iterout == 0) {
                /* nfiterw = i / iterout + 10;
                   nfiterh = i / iterout + 40; */
                G_debug(2, "iblock=%d i=%d miter=%d nwalk=%d nwalka=%d", iblock,
                        i, setup->miter, sim->nwalk, sim->nwalka);
            }

            if (sim->nwalka == 0 && i > 1)
                goto L_800;

            /* ************************************************************ */
            /*                               .... propagate one step */
            /* ************************************************************ */

            // the trapezoid rule could be removed (has very little effect),
            // still kept to not alter results
            // but factor (not addac) is used for infiltration
            addac = factor;
            if (i == 1) {
                addac = factor * .5;
            }
            nwalka = 0;
            sim->nstack = 0;

#pragma omp parallel private(l, k) reduction(+ : nwalka)
            {
#pragma omp for
                for (lw = 0; lw < sim->nwalk; lw++) {
                    if (sim->w[lw].m > EPS) { /* check the walker weight */
                        ++(nwalka);
                        l = (int)((sim->w[lw].x + stxm) / geometry->stepx) -
                            geometry->mx - 1;
                        k = (int)((sim->w[lw].y + stym) / geometry->stepy) -
                            geometry->my - 1;

                        if (l > geometry->mx - 1 || k > geometry->my - 1 ||
                            k < 0 || l < 0) {

                            G_debug(2, " k,l=%d,%d", k, l);
                            printf("    lw,w=%d %f %f", lw, sim->w[lw].y,
                                   sim->w[lw].m);
                            G_debug(2, "    stxym=%f %f", stxm, stym);
                            printf("    step=%f %f", geometry->stepx,
                                   geometry->stepy);
                            G_debug(2, "    m=%d %d", geometry->my,
                                    geometry->mx);
                            printf("    nwalka,nwalk=%d %d", sim->nwalka,
                                   sim->nwalk);
                            G_debug(2, "  ");
                        }

                        if (grids->zz[k][l] != UNDEF) {
                            if (grids->inf[k][l] != UNDEF &&
                                grids->inf[k][l] > 0) {
                                // Walker's contribution to water depth in this
                                // cell for this timestep [m]
                                double decr = factor * sim->w[lw].m;
                                // Compare with the depth the cell can absorb
                                // this timestep [m]
                                if (grids->inf[k][l] * setup->deltap > decr) {
                                    // The cell can absorb the full walker.
                                    // Reduce infiltration rate [m/s].
                                    grids->inf[k][l] -= decr / setup->deltap;
                                    // Eliminate the walker
                                    sim->w[lw].m = 0.;
                                    continue;
                                }
                                else {
                                    // The cell can't absorb the full walker.
                                    // Reduce the walker mass by the equivalent
                                    // of what an infiltration-rate source would
                                    // generate as walker weight.
                                    sim->w[lw].m -= sim->rwalk *
                                                    grids->inf[k][l] /
                                                    setup->sisum;
                                    // Cell's infiltration capacity is fully
                                    // exhausted
                                    grids->inf[k][l] = 0.;
                                    // Eliminate walker if needed
                                    if (sim->w[lw].m < 0.) {
                                        sim->w[lw].m = 0.;
                                        continue;
                                    }
                                }
                            }

                            grids->gama[k][l] +=
                                (addac * sim->w[lw].m); /* add walker weigh to
                                                      water depth or conc. */

                            double d1 = grids->gama[k][l] * conn;
                            double gaux, gauy;
                            gasdev(&sim->streams[lw], &gaux, &gauy);
                            double hhc = pow(d1, 3. / 5.);
                            double velx, vely;
                            if (hhc > settings->hhmax &&
                                inputs->wdepth == NULL) { /* increased diffusion
                                                     if w.depth > hhmax */
                                grids->dif[k][l] =
                                    (settings->halpha + 1) * deldif;
                                velx = sim->vavg[lw].x;
                                vely = sim->vavg[lw].y;
                            }
                            else {
                                grids->dif[k][l] = deldif;
                                velx = grids->v1[k][l];
                                vely = grids->v2[k][l];
                            }

                            if (inputs->traps != NULL &&
                                grids->trap[k][l] != 0.) { /* traps */

                                float eff = G_random_double(
                                    &sim->streams[lw]); /* random generator */

                                if (eff <= grids->trap[k][l]) {
                                    velx = -0.1 *
                                           grids->v1[k][l]; /* move it slightly
                                                               back */
                                    vely = -0.1 * grids->v2[k][l];
                                }
                            }

                            sim->w[lw].x +=
                                (velx +
                                 grids->dif[k][l] * gaux); /* move the walker */
                            sim->w[lw].y += (vely + grids->dif[k][l] * gauy);

                            if (hhc > settings->hhmax &&
                                inputs->wdepth == NULL) {
                                sim->vavg[lw].x =
                                    settings->hbeta *
                                    (sim->vavg[lw].x + grids->v1[k][l]);
                                sim->vavg[lw].y =
                                    settings->hbeta *
                                    (sim->vavg[lw].y + grids->v2[k][l]);
                            }

                            if (sim->w[lw].x <= geometry->xmin ||
                                sim->w[lw].y <= geometry->ymin ||
                                sim->w[lw].x >= geometry->xmax ||
                                sim->w[lw].y >= geometry->ymax) {
                                sim->w[lw].m = 1e-10; /* eliminate walker if it
                                                    is out of area */
                            }
                            else {
                                if (inputs->wdepth != NULL) {
                                    l = (int)((sim->w[lw].x + stxm) /
                                              geometry->stepx) -
                                        geometry->mx - 1;
                                    k = (int)((sim->w[lw].y + stym) /
                                              geometry->stepy) -
                                        geometry->my - 1;
                                    sim->w[lw].m *= grids->sigma[k][l];
                                }

                            } /* else */
                        } /*DEFined area */
                        else {
                            sim->w[lw].m = 1e-10; /* eliminate walker if it is
                                                out of area */
                        }
                    }
                } /* lw loop */
            }
            /* Total remaining walkers for this iteration */
            sim->nwalka = nwalka;

            /* Changes made by Soeren 8. Mar 2011 to replace the site walker
             * output implementation */
            /* Save all walkers located within the computational region and with
               valid z coordinates */
            if (outputs->outwalk != NULL &&
                (i == setup->miter ||
                 (setup->iterout > 0 && i % setup->iterout == 0))) {
                sim->nstack = 0;

                for (lw = 0; lw < sim->nwalk; lw++) {
                    /* Compute the  elevation raster map index */
                    l = (int)((sim->w[lw].x + stxm) / geometry->stepx) -
                        geometry->mx - 1;
                    k = (int)((sim->w[lw].y + stym) / geometry->stepy) -
                        geometry->my - 1;

                    /* Check for correct elevation raster map index */
                    if (l < 0 || l >= geometry->mx || k < 0 ||
                        k >= geometry->my)
                        continue;

                    if (sim->w[lw].m > EPS && grids->zz[k][l] != UNDEF) {

                        /* Save the 3d position of the walker */
                        sim->stack[sim->nstack].x =
                            geometry->mixx / geometry->conv +
                            sim->w[lw].x / geometry->conv;
                        sim->stack[sim->nstack].y =
                            geometry->miyy / geometry->conv +
                            sim->w[lw].y / geometry->conv;
                        sim->stack[sim->nstack].m = grids->zz[k][l];

                        sim->nstack++;
                    }
                } /* lw loop */
            }

            if (settings->ts && setup->iterout > 0 && i % setup->iterout == 0) {
                /* call output for iteration output */
                if (outputs->erdep != NULL)
                    erod(grids->gama, setup, geometry,
                         grids); /* divergence of gama field */

                int itime = (int)(i * setup->deltap * setup->timec);
                int ii = output_data(itime, conn, setup, geometry, settings,
                                     sim, inputs, outputs, grids);
                if (ii != 1)
                    G_fatal_error(_("Unable to write raster maps"));
            }

            /* Write the water depth each time step at an observation point */
            if (points->is_open) {
                double value = 0.0;
                int p;

                fprintf(points->output, "%.6d ", i);
                /* Write for each point */
                for (p = 0; p < points->npoints; p++) {
                    l = (int)((points->x[p] - geometry->mixx + stxm) /
                              geometry->stepx) -
                        geometry->mx - 1;
                    k = (int)((points->y[p] - geometry->miyy + stym) /
                              geometry->stepy) -
                        geometry->my - 1;

                    if (grids->zz[k][l] != UNDEF) {

                        if (inputs->wdepth == NULL)
                            value = geometry->step * grids->gama[k][l] *
                                    grids->cchez[k][l];
                        else
                            value = grids->gama[k][l] * grids->slope[k][l];

                        fprintf(points->output, "%2.4f ", value);
                    }
                    else {
                        /* Point is invalid, so a negative value is written */
                        fprintf(points->output, "%2.4f ", -1.0);
                    }
                }
                fprintf(points->output, "\n");
            }
        } /* miter */

    L_800:
        /* Soeren 8. Mar 2011: Why is this commented out? */
        /*        if (iwrib != nblock) {
           icount = icoub / iwrib;

           if (icoub == (icount * iwrib)) {
           ++icfl;
           nflw = icfl + 50;
           conn = (double) nblock / (double) iblock;

           }
           } */

        // Per-block sample for the Monte Carlo standard-deviation estimator
        // over nblock replicas, matching the original Fortran: accumulate
        // (gama * conn)^2 here, then finalize as sqrt(|gammas/nblock - gama^2|)
        // after the iblock loop closes. With nblock = 1 there is only one
        // sample and the finalized value is zero everywhere; the map becomes
        // meaningful once nblocks > 1 will be allowed.
        if (outputs->err != NULL) {
            for (k = 0; k < geometry->my; k++) {
                for (l = 0; l < geometry->mx; l++) {
                    if (grids->zz[k][l] != UNDEF) {
                        double d1 = grids->gama[k][l] * (double)conn;
                        grids->gammas[k][l] += d1 * d1;
                    } /* DEFined area */
                }
            }
        }
        if (outputs->erdep != NULL)
            erod(grids->gama, setup, geometry, grids);
    }
    /*                       ........ end of iblock loop */

    // Finalize the err map as the sample standard deviation of the per-block
    // estimators of the final field: sqrt(|E[X^2] - E[X]^2|), where each X
    // is gama * (nblock/iblock) recorded at the end of block iblock.
    if (outputs->err != NULL) {
        for (k = 0; k < geometry->my; k++) {
            for (l = 0; l < geometry->mx; l++) {
                if (grids->zz[k][l] != UNDEF) {
                    double mean_sq = grids->gama[k][l] * grids->gama[k][l];
                    double mean_of_sq = grids->gammas[k][l] / (double)nblock;
                    grids->gammas[k][l] = sqrt(fabs(mean_of_sq - mean_sq));
                }
            }
        }
    }

    /* Write final maps here because we know the last time stamp here */
    if (!settings->ts) {
        // All blocks have completed; gama is the eventual cumulative total,
        // so no extrapolation is needed.
        conn = 1.0;
        int itime = (int)(i * setup->deltap * setup->timec);
        int ii = output_data(itime, conn, setup, geometry, settings, sim,
                             inputs, outputs, grids);
        if (ii != 1)
            G_fatal_error(_("Cannot write raster maps"));
    }
    /* Close the observation logfile */
    if (points->is_open)
        fclose(points->output);

    points->is_open = 0;
}
