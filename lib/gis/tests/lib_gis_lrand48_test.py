"""Regression tests for the library drand48-family PRNG

These tests pin the exact output sequence of G_lrand48, G_mrand48 and
G_drand48 for a set of fixed seeds. Any change to the generator that alters
the produced numbers (e.g. a future reimplementation of the internal state)
is caught here, which makes this a guard for output compatibility.

The reference values below were captured from the current implementation.

The thread-safety test checks that the shared generator, drawn from by
several threads at once, still hands out exactly the single-threaded
sequence.

The tests at the end instead cover the caller-owned generator. Its
sequences need no reference values of their own: stream 0 is compared with
the shared generator, and the start of every other stream with an
independent computation of where the stream should begin. The remaining
tests check that a stream depends only on its seed, index and count, and
not on the presence of other threads.
"""

import subprocess
import sys
import threading
from ctypes import byref

import pytest

from grass.lib.gis import (
    G_drand48,
    G_lrand48,
    G_mrand48,
    G_random_double,
    G_random_seed,
    G_random_seed_stream,
    G_srand48,
    struct_G_random_state,
)

# First ten outputs of each generator after G_srand48(seed). The seeds cover
# 0, 1, two ordinary values, and the largest 32-bit seed value. Matching the
# first output verifies the seeding and the second one the state advancement;
# the remaining outputs and seeds guard against errors that occur only for
# some state values (e.g., a mishandled overflow), so a few of each suffice.
# The drand48 values are compared exactly on purpose: the generator state is
# a 48-bit integer, the returned value is state / 2**48, and a double
# represents both without rounding. The literals below were written by
# Python's repr, which produces the shortest decimal that parses back to
# exactly the double the implementation returned.
REFERENCE = {
    0: {
        "lrand48": [
            366850414,
            1610402240,
            206956554,
            1869309841,
            1239749840,
            1687491058,
            1486475625,
            791919534,
            1876694714,
            1600079540,
        ],
        "mrand48": [
            733700828,
            -1074162815,
            413913109,
            -556347614,
            -1815467615,
            -919985179,
            -1322016045,
            1583839069,
            -541577867,
            -1094808216,
        ],
        "drand48": [
            0.17082803610628972,
            0.7499019804849638,
            0.09637165562356742,
            0.8704652270270756,
            0.5773035067951078,
            0.785799258839674,
            0.6921941534586402,
            0.36876626992042105,
            0.8739040768618089,
            0.745095098450065,
        ],
    },
    1: {
        "lrand48": [
            89400484,
            976015093,
            1792756325,
            721524505,
            1214379247,
            3794415,
            402845420,
            2126940991,
            1611680321,
            786566648,
        ],
        "mrand48": [
            178800969,
            1952030186,
            -709454646,
            1443049011,
            -1866208802,
            7588830,
            805690840,
            -41085314,
            -1071606654,
            1573133297,
        ],
        "drand48": [
            0.041630344771878214,
            0.45449244472862915,
            0.8348172181669149,
            0.33598603014520023,
            0.5654894035661364,
            0.001766912391744313,
            0.18758951699996018,
            0.9904340799376641,
            0.7504971332295192,
            0.36627363815273384,
        ],
    },
    42: {
        "lrand48": [
            1598855263,
            735945821,
            238553827,
            906966006,
            174184913,
            1839192415,
            1071163602,
            1028245859,
            1483508427,
            1792276465,
        ],
        "mrand48": [
            -1097256770,
            1471891643,
            477107655,
            1813932012,
            348369827,
            -616582465,
            2142327205,
            2056491719,
            -1327950441,
            -710414366,
        ],
        "drand48": [
            0.7445250000610066,
            0.342701478718908,
            0.11108528244416149,
            0.422338957988309,
            0.08111117117831057,
            0.856440708026625,
            0.4987994221940788,
            0.4788142906446282,
            0.6908124443056387,
            0.8345937659621541,
        ],
    },
    1337: {
        "lrand48": [
            930965776,
            1690826993,
            854889137,
            583640949,
            1679004699,
            1147941803,
            76869624,
            1156695387,
            1887252525,
            560069492,
        ],
        "mrand48": [
            1861931553,
            -913313310,
            1709778274,
            1167281899,
            -936957898,
            -1999083690,
            153739248,
            -1981576522,
            -520462246,
            1120138985,
        ],
        "drand48": [
            0.4335147219981117,
            0.7873526742655201,
            0.3980887760791454,
            0.2717789959596715,
            0.7818474896603966,
            0.5345520579576117,
            0.03579520820343518,
            0.5386282629743491,
            0.8788204404903901,
            0.260802680918232,
        ],
    },
    2147483647: {
        "lrand48": [
            1718042167,
            1171047564,
            1842382256,
            1943353352,
            191378610,
            149962230,
            1496364007,
            530639902,
            1067967284,
            1339850607,
        ],
        "mrand48": [
            -858882961,
            -1952872168,
            -610202784,
            -408260591,
            382757220,
            299924460,
            -1302239282,
            1061279804,
            2135934568,
            -1615266081,
        ],
        "drand48": [
            0.8000257274407012,
            0.5453115162412985,
            0.8579260930802199,
            0.904944423908951,
            0.08911761002407914,
            0.06983160528760379,
            0.6967987899173202,
            0.24709845990317802,
            0.4973110204940987,
            0.6239165587473963,
        ],
    },
}


@pytest.mark.parametrize("seed", sorted(REFERENCE))
def test_lrand48_sequence_matches_reference(seed):
    """G_lrand48 reproduces the reference sequence for a fixed seed."""
    expected = REFERENCE[seed]["lrand48"]
    G_srand48(seed)
    assert [G_lrand48() for _ in range(len(expected))] == expected


@pytest.mark.parametrize("seed", sorted(REFERENCE))
def test_mrand48_sequence_matches_reference(seed):
    """G_mrand48 reproduces the reference sequence for a fixed seed."""
    expected = REFERENCE[seed]["mrand48"]
    G_srand48(seed)
    assert [G_mrand48() for _ in range(len(expected))] == expected


@pytest.mark.parametrize("seed", sorted(REFERENCE))
def test_drand48_sequence_matches_reference(seed):
    """G_drand48 reproduces the reference sequence for a fixed seed."""
    expected = REFERENCE[seed]["drand48"]
    G_srand48(seed)
    assert [G_drand48() for _ in range(len(expected))] == expected


def test_srand48_is_reproducible():
    """Re-seeding restarts the same sequence."""
    G_srand48(1337)
    first = [G_lrand48() for _ in range(20)]
    G_srand48(1337)
    second = [G_lrand48() for _ in range(20)]
    assert first == second


@pytest.mark.parametrize(
    ("negative", "equivalent"), [(-1, 4294967295), (-2147483648, 2147483648)]
)
def test_srand48_negative_seed_wraps_to_32_bits(negative, equivalent):
    """A negative seed seeds like its value modulo 2^32.

    Only the low 32 bits of the seed reach the generator, so -1 seeds like
    the largest 32-bit value, whose sequence is pinned in REFERENCE. This
    guards the seeding code, which callers passing a long rely on.
    """
    G_srand48(negative)
    first = [G_lrand48() for _ in range(20)]
    G_srand48(equivalent)
    second = [G_lrand48() for _ in range(20)]
    assert first == second


@pytest.mark.parametrize(
    "generate", [G_lrand48, G_mrand48, G_drand48], ids=["lrand48", "mrand48", "drand48"]
)
def test_shared_generator_is_thread_safe(generate):
    """Threads together consume exactly the single-threaded sequence.

    The shared generator serializes its state updates, so the threads
    must between them receive every value of the sequence once; only which
    thread gets which value depends on scheduling. Sorting removes that
    order before comparing. A set comparison would not do, because a
    correct sequence can contain duplicates. ctypes releases the GIL
    around each call, so the threads do run the C code concurrently.
    """
    seed = 1337
    num_threads = 4
    per_thread = 2500

    G_srand48(seed)
    serial = [generate() for _ in range(num_threads * per_thread)]

    threaded = [None] * num_threads

    def worker(index):
        threaded[index] = [generate() for _ in range(per_thread)]

    G_srand48(seed)
    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(num_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(value for values in threaded for value in values) == sorted(serial)


# The stream count used where a test needs some split but does not care
# which one.
NSTREAMS = 4096


def random_stream(seed, index, count, n):
    """Draw n values from stream index of the count streams derived from seed."""
    state = struct_G_random_state()
    G_random_seed_stream(byref(state), seed, index, count)
    return [G_random_double(byref(state)) for _ in range(n)]


def test_random_is_reproducible():
    """The same seed, index and count give the same sequence."""
    assert random_stream(1337, 0, NSTREAMS, 20) == random_stream(1337, 0, NSTREAMS, 20)


def test_random_range():
    """Generated values lie in [0, 1)."""
    assert all(0.0 <= value < 1.0 for value in random_stream(42, 3, NSTREAMS, 1000))


def test_random_streams_differ():
    """Streams derived from one seed do not repeat each other.

    Consecutive stream indices are the case a caller is most likely to
    use, so checking that their first values are all distinct guards the
    step that spaces the streams out along the generator cycle.
    """
    firsts = [random_stream(1337, index, NSTREAMS, 1)[0] for index in range(64)]
    assert len(set(firsts)) == len(firsts)


@pytest.mark.parametrize("seed", sorted(REFERENCE))
def test_random_stream_zero_matches_shared_generator(seed):
    """Stream 0 continues to produce what the shared generator produces.

    This is what lets code switch from G_drand48() to a caller-owned
    generator without changing its single-threaded results.
    """
    G_srand48(seed)
    shared = [G_drand48() for _ in range(100)]
    assert random_stream(seed, 0, NSTREAMS, 100) == shared


@pytest.mark.parametrize("seed", [0, 1337])
def test_random_seed_is_stream_zero(seed):
    """G_random_seed() gives stream 0, whatever the split into streams.

    Stream 0 starts at the seed state however many streams the cycle is
    split into, so a caller using a single stream can seed without
    choosing a count.
    """
    state = struct_G_random_state()
    G_random_seed(byref(state), seed)
    single = [G_random_double(byref(state)) for _ in range(100)]
    assert single == random_stream(seed, 0, 1, 100)
    assert single == random_stream(seed, 0, NSTREAMS, 100)


def test_random_streams_do_not_run_into_each_other():
    """Neighbouring streams stay apart over a long run.

    The streams are stretches of one cycle, so the guarantee that they do
    not overlap rests on the spacing between their starting points. This
    is a sanity check over a short run; the exact spacing is checked by
    test_random_streams_are_evenly_spaced.
    """
    n = 200000
    first = random_stream(1337, 0, NSTREAMS, n)
    second = random_stream(1337, 1, NSTREAMS, n)
    assert not set(first) & set(second)


# The generator constants, as in lrand48.c. Changing the generator or the
# way its cycle is split into streams changes the numbers of every stream
# but stream 0, so the tests below are meant to fail when that happens.
LCG_A = 0x5DEECE66D
LCG_B = 0xB
LCG_MODULUS = 2**48

# Splits to test: a power of two which divides the period, a count which
# does not divide it, a count beyond the rows of a large raster, and the
# largest count, which spaces the streams one step apart.
STREAM_COUNTS = [4096, 3, 100000, 2**48]


def lcg_jump_reference(state, steps):
    """Generator state after the given number of steps, in closed form

    Steps compose to a^n * x + b * (a^n - 1) / (a - 1) modulo 2^48. The
    division is exact over the integers but a - 1 has no inverse modulo
    2^48, so a^n is computed modulo 2^48 * (a - 1), which keeps the
    quotient correct modulo 2^48. Unlike the library, which composes the
    affine map by repeated squaring, this takes the additive term in
    closed form and reuses none of the library's code.
    """
    power = pow(LCG_A, steps, LCG_MODULUS * (LCG_A - 1))
    return (power * state + LCG_B * ((power - 1) // (LCG_A - 1))) % LCG_MODULUS


def random_states(seed, index, count, n):
    """Generator states behind the values of random_stream()

    A value is its state divided by 2^48, which a double holds exactly, so
    the multiplication recovers the state without rounding.
    """
    return [int(value * LCG_MODULUS) for value in random_stream(seed, index, count, n)]


def seed_state(seed):
    """Generator state which seeding with the given value produces"""
    return ((seed & 0xFFFFFFFF) << 16) | 0x330E


def test_lcg_jump_reference_matches_stepping():
    """The closed form agrees with drawing from the generator step by step.

    This ties the reference used below to the actual generator.
    """
    seed = 1337
    expected = [lcg_jump_reference(seed_state(seed), n) for n in range(1, 1001)]
    assert random_states(seed, 0, 1, 1000) == expected


@pytest.mark.parametrize("seed", [0, 1337, -1])
@pytest.mark.parametrize("count", STREAM_COUNTS)
def test_random_streams_are_evenly_spaced(seed, count):
    """A stream starts exactly its index times the stride along the cycle.

    The stride is the period divided by the count, rounded down, so all
    the streams fit into one cycle without wrapping around to stream 0.
    This is what makes them disjoint, and it is why an index equal to the
    count must be rejected: that stream would start a full cycle, or just
    short of one, after stream 0 and repeat its values. The first, a
    middle and the last stream are checked; the last one also shows that
    the highest index is accepted.
    """
    stride = LCG_MODULUS // count
    for index in sorted({1, count // 2, count - 1}):
        expected = lcg_jump_reference(seed_state(seed), index * stride + 1)
        assert random_states(seed, index, count, 1) == [expected], index


SEED_STREAM_SCRIPT = """
import sys
from ctypes import byref

from grass.lib.gis import G_random_seed_stream, struct_G_random_state

state = struct_G_random_state()
G_random_seed_stream(byref(state), 1337, int(sys.argv[1]), int(sys.argv[2]))
"""


@pytest.mark.parametrize(
    ("index", "count", "message"),
    [
        (NSTREAMS, NSTREAMS, "out of range"),
        (0, 0, "must be positive"),
        (0, 2**48 + 1, "period"),
    ],
    ids=["index_past_the_end", "no_streams", "more_streams_than_period"],
)
def test_random_seed_stream_rejects_impossible_stream(
    xy_session_for_module, tmp_path, index, count, message
):
    """An index or count which does not describe a stream is a fatal error.

    Runs in a subprocess because a fatal error exits the calling process,
    and with a session environment because without GISBASE the error
    message is not printed.
    """
    script = tmp_path / "seed_stream.py"
    script.write_text(SEED_STREAM_SCRIPT, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), str(index), str(count)],
        env=xy_session_for_module.env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0, "impossible stream was accepted"
    assert message in result.stderr


def test_random_seeds_differ():
    """The same stream index under different seeds gives different values."""
    assert random_stream(1, 7, NSTREAMS, 10) != random_stream(2, 7, NSTREAMS, 10)


def test_random_independent_of_shared_generator():
    """Drawing from the shared generator does not disturb a caller-owned one."""
    expected = random_stream(1337, 5, NSTREAMS, 10)

    state = struct_G_random_state()
    G_random_seed_stream(byref(state), 1337, 5, NSTREAMS)
    G_srand48(99)
    interleaved = []
    for _ in range(10):
        G_lrand48()
        interleaved.append(G_random_double(byref(state)))

    assert interleaved == expected


def test_random_unaffected_by_threading():
    """Each stream yields the same values whether or not threads are used.

    This is the property the shared generator cannot offer: results depend
    only on the seed, the stream index and the count, never on how many
    threads run or how they interleave. Each thread takes one of as many
    streams as there are threads, as a caller would.
    """
    seed = 1337
    num_streams = 8
    n = 5000

    serial = [
        random_stream(seed, index, num_streams, n) for index in range(num_streams)
    ]

    threaded = [None] * num_streams

    def worker(index):
        threaded[index] = random_stream(seed, index, num_streams, n)

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(num_streams)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert threaded == serial
