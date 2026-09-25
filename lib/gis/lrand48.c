/*!
 * \file lib/gis/lrand48.c
 *
 * \brief GIS Library - Pseudo-random number generation
 *
 * The generator is the standard drand48 linear congruential generator
 * X' = (A * X + B) mod 2^48 with A = 0x5DEECE66D and B = 0xB.
 *
 * Two interfaces are provided. The G_*rand48() functions draw from a
 * single generator shared by the whole program. When C11 atomic
 * operations are available, that shared state is advanced with an atomic
 * compare-and-swap and the generating functions are thread-safe: the
 * sequence of generated values for a given seed is the same as in a
 * single-threaded run. Which thread receives which value depends on
 * scheduling, so results are fully reproducible only with single-threaded
 * execution. Without C11 atomics (notably MSVC, which defines
 * __STDC_NO_ATOMICS__), the shared generator falls back to plain state
 * updates, so multi-threaded usage is safe only when compiled with C11
 * atomics. The seeding functions are not thread-safe; see G_srand48().
 *
 * The G_random_*() functions instead advance a generator owned by the
 * caller, one state per thread or per unit of work, never shared. These
 * need neither atomics nor locks, are safe on every build, and give each
 * stream a reproducible sequence of its own. This is what parallel code
 * that must produce the same result for a given seed regardless of the
 * number of threads should use: number the streams by the unit of work,
 * the row, the chunk of walkers or the run. The streams are disjoint
 * stretches of the same cycle the shared generator walks, split into as
 * many streams as the caller asks for, and stream 0 starts where
 * G_srand48() does. See \ref gislib_random_streams for the model and the
 * usage patterns.
 *
 * SPDX-FileCopyrightText: 2014-2026 GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * \authors Glynn Clements, Maris Nartiss (thread safety)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
    !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#define LRAND48_ATOMIC 1
#else
#define LRAND48_ATOMIC 0
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

#ifdef HAVE_GETTIMEOFDAY
#include <sys/time.h>
#else
#include <time.h>
#endif

#include <sys/types.h>
#include <unistd.h>

typedef unsigned int uint32;
typedef signed int int32;

#define LCG_A  UINT64_C(0x5DEECE66D)
#define LCG_B  UINT64_C(0xB)
#define MASK48 UINT64_C(0xFFFFFFFFFFFF)

/* Advance the generator by one step. The multiplication may wrap around
 * at 2^64; that does not change the result modulo 2^48. */
static inline unsigned long long lcg_step(unsigned long long cur)
{
    return (LCG_A * cur + LCG_B) & MASK48;
}

/* Turn a seed value into a generator state the way drand48 does. */
static inline unsigned long long lcg_seed(unsigned long long seed)
{
    return ((seed & 0xFFFFFFFF) << 16) | 0x330E;
}

/* The period of the generator: every state lies on one cycle of this
 * length, and caller-owned streams are stretches of it. */
#define LCG_PERIOD (MASK48 + 1)

/* Advance the generator by an arbitrary number of steps without taking
 * them one at a time. One step is the affine map x -> a * x + c, and
 * composing two such maps gives another, so the map for `steps` steps is
 * built by repeated squaring, as an integer power would be. */
static unsigned long long lcg_jump(unsigned long long x,
                                   unsigned long long steps)
{
    unsigned long long a_total = 1, c_total = 0; /* the identity map */
    unsigned long long a = LCG_A, c = LCG_B;     /* one step */

    while (steps) {
        if (steps & 1) {
            c_total = (a * c_total + c) & MASK48;
            a_total = (a * a_total) & MASK48;
        }
        /* Square the map, so a and c then describe twice as many steps. */
        c = (a * c + c) & MASK48;
        a = (a * a) & MASK48;
        steps >>= 1;
    }

    return (a_total * x + c_total) & MASK48;
}

#if LRAND48_ATOMIC
/* The whole 48-bit state is kept in one atomic integer so that it can be
 * advanced in one compare-and-swap: a successful swap is exactly one
 * generator step, giving the same sequence of states as a single-threaded
 * run. */
static _Atomic unsigned long long state;
#else
static unsigned long long state;
#endif

static int seeded;

/*!
 * \brief Seed the shared pseudo-random number generator
 *
 * This function is not thread-safe. In a multi-threaded program, call
 * `G_srand48()` once *before* starting the worker threads; it must not
 * run concurrently with another thread seeding or generating values.
 *
 * Only the low 32 bits of the seed reach the generator: seeds equal
 * modulo 2^32 give the same sequence, and a negative value seeds like its
 * two's complement low 32 bits. Seeds which differ only in their top bits
 * are not independent either: seeds 2^31 apart give values which differ
 * by exactly one half at every draw, and seeds 2^30 apart by one quarter.
 * Keep seeds below 2^30, and derive several independent sequences from
 * one seed with G_random_seed_stream() rather than from several seeds.
 *
 * \param[in] seedval seed, used modulo 2^32
 */
void G_srand48(long seedval)
{
#if LRAND48_ATOMIC
    atomic_store(&state, lcg_seed((unsigned long long)seedval));
#else
    state = lcg_seed((unsigned long long)seedval);
#endif
    seeded = 1;
}

/*!
 * \brief Seed the shared pseudo-random number generator from the time and PID
 *
 * A weak hash of the current time and PID is generated and used to
 * seed the PRNG
 *
 * This function is not thread-safe. In a multi-threaded program, call
 * `G_srand48_auto()` once *before* starting the worker threads; it must
 * not run concurrently with another thread seeding or generating values.
 *
 * A value of `GRASS_RANDOM_SEED` or `SOURCE_DATE_EPOCH` is used modulo
 * 2^32, like any seed; see G_srand48().
 *
 * \return generated seed value passed to G_srand48()
 */
long G_srand48_auto(void)
{
    unsigned long seed;
    char *grass_random_seed = getenv("GRASS_RANDOM_SEED");

    if (!grass_random_seed)
        grass_random_seed = getenv("SOURCE_DATE_EPOCH");
    if (grass_random_seed) {
        seed = strtoull(grass_random_seed, NULL, 10);
    }
    else {
        seed = (unsigned long)getpid();

#ifdef HAVE_GETTIMEOFDAY
        {
            struct timeval tv;

            if (gettimeofday(&tv, NULL) < 0)
                G_fatal_error(_("gettimeofday failed: %s"), strerror(errno));
            seed += (unsigned long)tv.tv_sec;
            seed += (unsigned long)tv.tv_usec;
        }
#else
        {
            time_t t = time(NULL);

            seed += (unsigned long)t;
        }
#endif
    }

    G_srand48((long)seed);
    return (long)seed;
}

/* Advance the shared generator by one step and return the new state.
 * Callers derive their result from the returned value, not from the
 * shared state, so concurrent calls each get a distinct step of the
 * sequence. */
static unsigned long long G__next(void)
{
    if (!seeded)
        G_fatal_error(_("Pseudo-random number generator not seeded"));

#if LRAND48_ATOMIC
    {
        unsigned long long cur =
            atomic_load_explicit(&state, memory_order_relaxed);
        unsigned long long next;

        do {
            next = lcg_step(cur);
        } while (!atomic_compare_exchange_weak_explicit(
            &state, &cur, next, memory_order_relaxed, memory_order_relaxed));

        return next;
    }
#else
    state = lcg_step(state);
    return state;
#endif
}

/*!
 * \brief Generate an integer in the range [0, 2^31)
 *
 * This function is thread-safe only when compiled with C11 atomics
 * (see the comment at the top of the file).
 *
 * \return the generated value
 */
long G_lrand48(void)
{
    return (long)(G__next() >> 17);
}

/*!
 * \brief Generate an integer in the range [-2^31, 2^31)
 *
 * This function is thread-safe only when compiled with C11 atomics
 * (see the comment at the top of the file).
 *
 * \return the generated value
 */
long G_mrand48(void)
{
    return (long)(int32)(uint32)(G__next() >> 16);
}

/*!
 * \brief Generate a floating-point value in the range [0,1)
 *
 * This function is thread-safe only when compiled with C11 atomics
 * (see the comment at the top of the file).
 *
 * \return the generated value
 */
double G_drand48(void)
{
    /* The state is below 2^53, so the conversion to double is exact. */
    return (double)G__next() / 281474976710656.0; /* 2^48 */
}

/*!
 * \brief Seed a caller-owned pseudo-random number generator
 *
 * With the same seed, the generator produces the sequence the shared
 * generator produces after G_srand48(), so code moving from the shared
 * generator to a caller-owned one reproduces its existing results.
 *
 * The caller owns the state, so this function is thread-safe as long as
 * no two threads seed the same state.
 *
 * The seed must lie between -2^31 and 2^32 - 1, the values the current
 * generator can tell apart; anything else is a fatal error rather than a
 * silent use of the low 32 bits. A negative seed gives the stream
 * G_srand48() gives for it, that of its two's complement 32-bit value.
 * Seeds 2^31 or 2^30 apart give values shifted by one half or one
 * quarter, see G_srand48(); to obtain several independent streams, derive
 * them from one seed with G_random_seed_stream() rather than from several
 * seeds.
 *
 * \param[out] state generator state to seed
 * \param[in] seed value to seed the generator with
 *
 * \return the number of values the generator produces before it repeats,
 *         see G_random_seed_stream()
 */
long long G_random_seed(struct G_random_state *state, long long seed)
{
    return G_random_seed_stream(state, seed, 0, 1);
}

/*!
 * \brief Seed one of several caller-owned generators derived from one seed
 *
 * Gives a unit of work its own stream: with one stream per raster row,
 * pass the row number and the number of rows; with one stream per chunk
 * of walkers, the chunk number and the number of chunks; with one stream
 * per run of an ensemble, the run number and the number of runs. Numbering
 * the streams by the unit of work rather than by the thread keeps results
 * independent of how the work is scheduled, and therefore of the number
 * of threads.
 *
 * A struct G_random_state is a plain variable holding one generator.
 * The program keeps one for every sequence it draws at a time, one per
 * thread when threads take rows in turn, one per chunk of work that
 * lasts the whole computation, one per run, and uses each from one
 * thread at a time. Such states share no memory, so drawing from them
 * needs no synchronization, and they still fit together: seeded with the
 * same seed and number of streams, they take disjoint parts of one
 * layout of the cycle, so their values do not collide, overlap or shift
 * each other by a constant at the same draw. Seeding puts a state at the
 * start of its stream whatever the state held before, so the same call
 * always restarts the same sequence. A state is seeded when its unit of
 * work starts drawing and not again while the unit lasts: a row's stream
 * continues across the row's columns, and in the same way a chunk's
 * stream continues across time steps. Where units are short and many,
 * as rows, each thread reuses one state and seeds it for every unit it
 * takes, at one jump of about 100 ns per unit; where a unit lasts the
 * whole computation, as a chunk, it keeps a state of its own. Since the
 * stream number is the unit's number, the values are fixed by the seed
 * and the layout, not by which thread draws them. Both patterns are
 * shown with examples in \ref gislib_random_streams.
 *
 * The streams start evenly spaced along the generator cycle, which is
 * split into \p streams parts, or \p streams + 1 parts when \p streams is
 * even, with an odd stride, so they are disjoint as long as each draws
 * fewer than the returned number of values, about period / (\p streams +
 * 1). The period of the current generator is 2^48 (about 2.8e14). The
 * number of parts and the stride are odd because two streams whose
 * starts are a multiple of 2^46 steps apart, a quarter of the cycle, are
 * one sequence up to a constant: one stream's values are the other's
 * plus a constant at every draw. An even split contains such pairs: with
 * 4096 parts, stream 2048 would start 2^47 steps after stream 0 and
 * give, for seed 42, 0.2445, 0.8427, 0.6111 where stream 0 gives 0.7445,
 * 0.3427, 0.1111, each value one half more. So does a stride which is a
 * power of two, or three times one: with 2^25 - 1 streams the stride
 * would be exactly 2^23, so stream 2^23 would start 2^46 steps after
 * stream 0 and give 0.9945, 0.5927, 0.3611, one quarter more; with the
 * stride rounded down to 8,388,607 it gives 0.2474, 0.2308, 0.7298. The
 * rounding costs one value per part. Both rules remove twins at the same
 * draw only; twins at a lag remain in every layout. With a stride s and
 * m * 2^46 = q * s + d, for m = 1, 2, 3 and 0 <= d < s, stream j + q is
 * stream j plus m / 4 at lag d, and stream j + q + 1 is stream j at lag
 * s - d the other way round. With 2^25 - 1 streams, stream 2^23 + 1
 * starts one step before the quarter mark, and its draw t + 1 is stream
 * 0's draw t plus one quarter. A twin at a lag matters only when a
 * stream is drawn past that lag. For counts far below 2^24 the lag is
 * about a quarter of the stride for m = 1 and 3 and a half for m = 2;
 * for counts around 2^24 and above it can be small, as in the example.
 * See \ref gislib_random_streams_odd.
 *
 * Stream 0 is what G_random_seed() gives, so code moving from the shared
 * generator to this one reproduces its single-threaded results with
 * stream 0.
 *
 * The caller owns the state, so this function is thread-safe as long as
 * no two threads seed the same state.
 *
 * A \p seed outside -2^31 to 2^32 - 1, a number of \p streams which is
 * not positive or not below the period, or a \p stream outside 0 to
 * \p streams - 1, is a fatal error.
 *
 * The returned length is what the caller can compare with the most
 * values it is going to draw from the stream, for example columns times
 * draws per cell for a row, and refuse or warn when the stream is too
 * short for its layout. Nothing is enforced, since the library cannot
 * know how many values a stream will draw; a stream drawn past its
 * length continues into the next stream's part and from there produces
 * that stream's values, shifted in time. The number of streams times
 * the values each draws must therefore stay below 2^48. A million streams
 * are 281 million values long each, ten million streams 28 million, a
 * billion streams 281 thousand, and a hundred billion streams 2,813, so
 * a billion streams drawing ten thousand values each still fit and a
 * hundred billion do not.
 *
 * \param[out] state generator state to seed
 * \param[in] seed value to seed the generator with, see G_random_seed()
 * \param[in] stream number of this stream, from 0 to \p streams - 1, for
 *            example the row, chunk or run number
 * \param[in] streams number of streams derived from \p seed, for example
 *            the number of rows, chunks or runs, positive
 *
 * \return the number of values this stream produces before it reaches
 *         the next one, the period divided by the number of parts; a
 *         generator with a longer period than the return type can hold
 *         returns LLONG_MAX
 */
long long G_random_seed_stream(struct G_random_state *state, long long seed,
                               long long stream, long long streams)
{
    unsigned long long parts, stride;

    if (seed < -(1LL << 31) || seed > (1LL << 32) - 1)
        G_fatal_error(_("Random number seed %lld is outside the range from "
                        "-2147483648 to 4294967295 the generator can use"),
                      seed);
    if (streams <= 0)
        G_fatal_error(
            _("The number of random number streams must be positive, not %lld"),
            streams);
    if ((unsigned long long)streams >= LCG_PERIOD)
        G_fatal_error(_("Cannot derive %lld random number streams from one "
                        "seed (the number must be below the generator's "
                        "period of %llu)"),
                      streams, (unsigned long long)LCG_PERIOD);
    if (stream < 0 || stream >= streams)
        G_fatal_error(_("Random number stream %lld is out of range "
                        "(must be between 0 and %lld)"),
                      stream, streams - 1);

    /* An odd number of parts and an odd stride keep every pair of streams
     * away from the power-of-two fractions of the cycle at which this
     * generator repeats itself up to a constant; see the description
     * above. The stride is rounded down to odd so that the parts still fit
     * into one cycle; a single stream keeps the whole period. The
     * arithmetic below is modular, so it is done in unsigned integers. */
    parts = (unsigned long long)streams | 1;
    stride = LCG_PERIOD / parts;
    if (parts > 1 && stride % 2 == 0)
        stride--;
    state->state = lcg_jump(lcg_seed((unsigned long long)seed),
                            (unsigned long long)stream * stride);

    return (long long)stride;
}

/*!
 * \brief Generate a floating-point value in the range [0,1) from a
 *        caller-owned generator
 *
 * Thread-safe as long as no two threads share a state. Unlike
 * G_drand48(), this needs no atomics and so behaves identically on every
 * build.
 *
 * \param[in,out] state generator state, seeded with G_random_seed() or
 *                G_random_seed_stream()
 *
 * \return the generated value
 */
double G_random_double(struct G_random_state *state)
{
    state->state = lcg_step(state->state);
    /* The state is below 2^53, so the conversion to double is exact. */
    return (double)state->state / 281474976710656.0; /* 2^48 */
}
