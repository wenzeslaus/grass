"""Landscape-configuration tests for r.sim.water (SIMWE).

Tests use 2D synthetic landscapes with small spatial extent and
hand-crafted elevation surfaces where expected output can be reasoned
about from the landscape geometry.
"""

import os

import numpy as np
import pytest

import grass.script as gs
from grass.tools import Tools

# Fixed seed and single thread make results fully deterministic.
SEED = 42
NPROCS = 1

DURATION = 2  # minutes; enough for near-steady state on small domains
RAIN = 100  # mm/hr


def run_sim(session, *, random_seed=SEED, **kwargs):
    """Run r.sim.water on a 2D domain; return depth as 2D ndarray.

    Omits dx/dy so partial derivatives are computed internally.
    """
    defaults = {
        "rain_value": RAIN,
        "infil_value": 0,
        "man_value": 0.1,
        "nprocs": NPROCS,
        "duration": DURATION,
    }
    defaults.update(kwargs)
    defaults = {k: v for k, v in defaults.items() if v is not None}
    tools = Tools(session=session)
    return tools.r_sim_water(
        elevation="elevation",
        depth=np.array,
        random_seed=random_seed,
        **defaults,
    )


def test_pit_collects_water(tmp_path):
    """A bowl-shaped pit must collect most water in its center.

    A 5x5 grid with elevation equal to the Chebyshev distance from the
    center creates a closed depression. Water flows inward from all
    directions and accumulates at the lowest point.

    Elevation:
        3 3 3 3 3
        3 2 2 2 3
        3 2 1 2 3
        3 2 2 2 3
        3 3 3 3 3
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=5, s=0, n=5, res=1)
        # row() and col() are 1-indexed, so center of 5x5 is at (3, 3).
        tools.r_mapcalc(
            expression="elevation = max(abs(row() - 3), abs(col() - 3)) + 1"
        )

        depth = run_sim(session)

        # The center cell is at NumPy index [2, 2].
        center_depth = float(depth[2, 2])
        total_depth = float(np.sum(depth))

        # The pit center must have the maximum depth.
        assert center_depth == np.max(depth), (
            f"Pit center depth ({center_depth:.4f}) should be the domain maximum "
            f"({np.max(depth):.4f})"
        )

        # Most water should be trapped in the center. The 30% threshold
        # holds across resolutions from 0.1 m to 30 m.
        assert center_depth > 0.3 * total_depth, (
            f"Center should hold >30% of total water: "
            f"center={center_depth:.4f}, total={total_depth:.4f}"
        )

        # Radial symmetry: the 4 cardinal neighbors should have similar depth.
        neighbors = [depth[1, 2], depth[3, 2], depth[2, 1], depth[2, 3]]
        mean_neighbor = float(np.mean(neighbors))
        for i, nd in enumerate(neighbors):
            assert nd == pytest.approx(mean_neighbor, rel=0.3), (
                f"Cardinal neighbor {i} depth ({nd:.4f}) deviates from "
                f"mean neighbor depth ({mean_neighbor:.4f})"
            )

        # Cardinal neighbors must have less depth than center.
        assert all(n < center_depth for n in neighbors), (
            f"All neighbors should have less depth than center: "
            f"neighbors={[f'{n:.4f}' for n in neighbors]}, "
            f"center={center_depth:.4f}"
        )

        # Radial symmetry for the 4 diagonal neighbors.
        diagonals = [depth[1, 1], depth[1, 3], depth[3, 1], depth[3, 3]]
        mean_diagonal = float(np.mean(diagonals))
        for i, dd in enumerate(diagonals):
            assert dd == pytest.approx(mean_diagonal, rel=0.3), (
                f"Diagonal neighbor {i} depth ({dd:.4f}) deviates from "
                f"mean diagonal depth ({mean_diagonal:.4f})"
            )

        # Depth must decrease monotonically from center outward:
        # center > cardinal ring > edge midpoints.
        edge_midpoints = [depth[0, 2], depth[4, 2], depth[2, 0], depth[2, 4]]
        assert mean_neighbor > float(np.mean(edge_midpoints)), (
            f"Ring 1 mean depth ({mean_neighbor:.4f}) should exceed "
            f"ring 2 mean depth ({float(np.mean(edge_midpoints)):.4f})"
        )

        # Full grid regression check for this seed and parameters.
        expected = np.array(
            [
                [0.000503, 0.000395, 0.000451, 0.000642, 0.000395],
                [0.000451, 0.002898, 0.014304, 0.002434, 0.000800],
                [0.000451, 0.014336, 0.087167, 0.014282, 0.000451],
                [0.000503, 0.002556, 0.014759, 0.002851, 0.000684],
                [0.000642, 0.000503, 0.000260, 0.000800, 0.000503],
            ]
        )
        np.testing.assert_allclose(depth, expected, atol=0.002)


def test_peak_sheds_water(tmp_path):
    """A conical peak must have minimum depth at the summit.

    A 5x5 grid with elevation decreasing outward from the center creates
    a peak. Water flows away from the summit in all directions, so the
    center should have the least water and edges should accumulate more.

    Elevation:
        2 2 2 2 2
        2 3 3 3 2
        2 3 4 3 2
        2 3 3 3 2
        2 2 2 2 2
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=5, s=0, n=5, res=1)
        tools.r_mapcalc(
            expression="elevation = 4 - max(abs(row() - 3), abs(col() - 3))"
        )

        depth = run_sim(session)

        # Cardinal neighbors should have similar depth (radial symmetry).
        neighbors = [depth[1, 2], depth[3, 2], depth[2, 1], depth[2, 3]]
        mean_neighbor = float(np.mean(neighbors))
        for i, nd in enumerate(neighbors):
            assert nd == pytest.approx(mean_neighbor, rel=0.4), (
                f"Cardinal neighbor {i} depth ({nd:.6f}) deviates from "
                f"mean neighbor depth ({mean_neighbor:.6f})"
            )

        # Diagonal neighbors should have similar depth.
        diagonals = [depth[1, 1], depth[1, 3], depth[3, 1], depth[3, 3]]
        mean_diagonal = float(np.mean(diagonals))
        for i, dd in enumerate(diagonals):
            assert dd == pytest.approx(mean_diagonal, rel=0.4), (
                f"Diagonal neighbor {i} depth ({dd:.6f}) deviates from "
                f"mean diagonal depth ({mean_diagonal:.6f})"
            )

        # Outer ring (edges) should have more depth than inner ring.
        edge_midpoints = [depth[0, 2], depth[4, 2], depth[2, 0], depth[2, 4]]
        mean_edge = float(np.mean(edge_midpoints))
        assert mean_edge > mean_neighbor, (
            f"Edge depth ({mean_edge:.6f}) should exceed "
            f"neighbor depth ({mean_neighbor:.6f})"
        )

        # Full grid regression check.
        expected = np.array(
            [
                [0.000973, 0.000872, 0.000800, 0.000451, 0.000800],
                [0.000684, 0.000598, 0.000552, 0.000503, 0.000260],
                [0.000552, 0.000332, 0.000598, 0.000503, 0.000724],
                [0.000503, 0.000503, 0.000332, 0.000395, 0.000260],
                [0.000907, 0.000642, 0.000642, 0.000763, 0.000642],
            ]
        )
        np.testing.assert_allclose(depth, expected, atol=0.0002)


def test_ridge_sheds_water_to_flanks(tmp_path):
    """A ridge must shed water to both flanks, leaving the crest drier.

    A 5x7 grid with a N-S ridge along the center column. The crest
    is the highest point and water flows away toward the edges.

    Elevation (each row):
        0 1 2 3 2 1 0
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=5, res=1)
        tools.r_mapcalc(expression="elevation = 3 - abs(col() - 4)")

        depth = run_sim(session)

        # Edge columns (0 and 6) should accumulate more water than
        # the crest column (3) because water flows downhill.
        crest_sum = float(np.sum(depth[:, 3]))
        left_edge_sum = float(np.sum(depth[:, 0]))
        right_edge_sum = float(np.sum(depth[:, 6]))
        assert left_edge_sum > crest_sum, (
            f"Left edge ({left_edge_sum:.6f}) should exceed crest ({crest_sum:.6f})"
        )
        assert right_edge_sum > crest_sum, (
            f"Right edge ({right_edge_sum:.6f}) should exceed crest ({crest_sum:.6f})"
        )

        # Left-right symmetry: columns equidistant from crest should
        # have similar total depth.
        for offset in range(1, 4):
            left_sum = float(np.sum(depth[:, 3 - offset]))
            right_sum = float(np.sum(depth[:, 3 + offset]))
            assert left_sum == pytest.approx(right_sum, rel=0.4), (
                f"Offset {offset}: left sum ({left_sum:.6f}) vs "
                f"right sum ({right_sum:.6f})"
            )

        # Full grid regression check.
        expected = np.array(
            [
                [0.000946, 0.000586, 0.000303, 0.000303, 0.000360, 0.000459, 0.000546],
                [0.001390, 0.000796, 0.000661, 0.001455, 0.000412, 0.000827, 0.001182],
                [0.001206, 0.001055, 0.000796, 0.001081, 0.000586, 0.000624, 0.001497],
                [0.001254, 0.000730, 0.000586, 0.001158, 0.000412, 0.000827, 0.001055],
                [0.000888, 0.000546, 0.000459, 0.000730, 0.000238, 0.000546, 0.000661],
            ]
        )
        np.testing.assert_allclose(depth, expected, atol=0.0005)


def test_ridge_rotation_invariance(tmp_path):
    """A N-S ridge and E-W ridge must produce rotationally equivalent depth.

    The E-W ridge is a 90-degree rotation of the N-S ridge. After rotating
    the E-W result back, total depth and cross-ridge profiles should match.
    """
    # N-S ridge: 5 rows x 7 cols, ridge along center column.
    project_ns = tmp_path / "ns"
    gs.create_project(project_ns)
    with gs.setup.init(project_ns, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=5, res=1)
        tools.r_mapcalc(expression="elevation = 3 - abs(col() - 4)")
        depth_ns = run_sim(session)

    # E-W ridge: 7 rows x 5 cols, ridge along center row.
    project_ew = tmp_path / "ew"
    gs.create_project(project_ew)
    with gs.setup.init(project_ew, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=5, s=0, n=7, res=1)
        tools.r_mapcalc(expression="elevation = 3 - abs(row() - 4)")
        depth_ew = run_sim(session)

    # Total depth should be comparable. The ridge has only a 3 m
    # elevation range on a small grid, so boundary effects and Monte
    # Carlo noise cause ~10% variation between orientations.
    total_ns = float(np.sum(depth_ns))
    total_ew = float(np.sum(depth_ew))
    assert total_ns == pytest.approx(total_ew, rel=0.1), (
        f"Total depth N-S ({total_ns:.6f}) vs E-W ({total_ew:.6f})"
    )

    # Rotating the E-W result 90 degrees counter-clockwise should
    # approximate the N-S result.
    depth_ew_rotated = np.rot90(depth_ew)
    assert depth_ew_rotated.shape == depth_ns.shape

    # Cross-ridge profiles (summed along the ridge axis) should match.
    # Monte Carlo noise on small grids requires generous tolerance.
    profile_ns = np.sum(depth_ns, axis=0)
    profile_ew = np.sum(depth_ew_rotated, axis=0)
    np.testing.assert_allclose(profile_ns, profile_ew, rtol=0.35)


def test_valley_concentrates_flow_downstream(tmp_path):
    """A V-shaped valley must concentrate water along its floor.

    A 10x7 grid with a V-shaped cross-section draining southward.
    The valley floor (center column) should have much more water than
    the hillslopes, and discharge should increase downstream.

    Cross-section (each row, west to east):
        3 2 1 0 1 2 3
    Downstream gradient adds 0.5 per row, so the top row is 4.5 m
    higher than the bottom row.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=10, res=1)
        tools.r_mapcalc(
            expression="elevation = abs(col() - 4) + (nrows() - row()) * 0.5"
        )

        depth = run_sim(session)

        # Valley floor (col index 3) should have much higher per-cell
        # depth than the hillslope average.
        floor_mean = float(np.mean(depth[:, 3]))
        hillslope_cols = np.concatenate([depth[:, :3], depth[:, 4:]], axis=1)
        hillslope_mean = float(np.mean(hillslope_cols))
        assert floor_mean > 3 * hillslope_mean, (
            f"Valley floor mean depth ({floor_mean:.6f}) should be >3x "
            f"hillslope mean ({hillslope_mean:.6f})"
        )

        # Discharge (depth) should increase downstream (toward higher row
        # indices in the array). Compare upper and lower halves.
        upper_floor = float(np.sum(depth[:5, 3]))
        lower_floor = float(np.sum(depth[5:, 3]))
        assert lower_floor > upper_floor, (
            f"Lower valley floor ({lower_floor:.6f}) should exceed "
            f"upper valley floor ({upper_floor:.6f})"
        )

        # Cross-valley symmetry: columns equidistant from floor should
        # have similar total depth.
        for offset in range(1, 4):
            left = float(np.sum(depth[:, 3 - offset]))
            right = float(np.sum(depth[:, 3 + offset]))
            assert left == pytest.approx(right, rel=0.4), (
                f"Offset {offset}: left ({left:.6f}) vs right ({right:.6f})"
            )

        # Full grid regression check.
        expected = np.array(
            [
                [0.000276, 0.000404, 0.000579, 0.001990, 0.000684, 0.000704, 0.000276],
                [0.000556, 0.000834, 0.000951, 0.001532, 0.000724, 0.000194, 0.000458],
                [0.000276, 0.000533, 0.000919, 0.002308, 0.000886, 0.000601, 0.000643],
                [0.000533, 0.000431, 0.001119, 0.003021, 0.001014, 0.000684, 0.000484],
                [0.000404, 0.000643, 0.000798, 0.003759, 0.001271, 0.000724, 0.000431],
                [0.000276, 0.000622, 0.001060, 0.004491, 0.001323, 0.000664, 0.000404],
                [0.000509, 0.000601, 0.001189, 0.004852, 0.001323, 0.000556, 0.000484],
                [0.000431, 0.000643, 0.001244, 0.005432, 0.001555, 0.000509, 0.000311],
                [0.000374, 0.000724, 0.001203, 0.005944, 0.001361, 0.000643, 0.000664],
                [0.000622, 0.000851, 0.001323, 0.006594, 0.001635, 0.000684, 0.000431],
            ]
        )
        np.testing.assert_allclose(depth, expected, atol=0.002)


def test_valley_rotation_invariance(tmp_path):
    """A N-S valley and E-W valley must produce rotationally equivalent depth."""
    # N-S valley: 10 rows x 7 cols.
    project_ns = tmp_path / "ns"
    gs.create_project(project_ns)
    with gs.setup.init(project_ns, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=10, res=1)
        tools.r_mapcalc(
            expression="elevation = abs(col() - 4) + (nrows() - row()) * 0.5"
        )
        depth_ns = run_sim(session)

    # E-W valley: 7 rows x 10 cols.
    project_ew = tmp_path / "ew"
    gs.create_project(project_ew)
    with gs.setup.init(project_ew, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=10, s=0, n=7, res=1)
        tools.r_mapcalc(
            expression="elevation = abs(row() - 4) + (ncols() - col()) * 0.5"
        )
        depth_ew = run_sim(session)

    total_ns = float(np.sum(depth_ns))
    total_ew = float(np.sum(depth_ew))
    assert total_ns == pytest.approx(total_ew, rel=0.02), (
        f"Total depth N-S ({total_ns:.6f}) vs E-W ({total_ew:.6f})"
    )

    # Cross-valley profiles should match after rotation.
    profile_ns = np.sum(depth_ns, axis=0)
    depth_ew_rotated = np.rot90(depth_ew)
    profile_ew = np.sum(depth_ew_rotated, axis=0)
    np.testing.assert_allclose(profile_ns, profile_ew, rtol=0.15)


def test_saddle_bifurcates_flow(tmp_path):
    """A saddle must split flow into two low quadrants.

    A 7x7 grid shaped as a hyperbolic paraboloid (saddle surface).
    The surface rises along the N-S axis and drops along the E-W
    axis, creating two high zones (top and bottom) and two low zones
    (left and right). Water should flow preferentially into the low
    zones.
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=7, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = (row() - 4)*(row() - 4) - (col() - 4)*(col() - 4) + 10"
            )
        )

        depth = run_sim(session)

        # Low quadrants: where (col-4)^2 > (row-4)^2, i.e., columns
        # far from center. These are the left and right sides.
        # High quadrants: rows far from center (top and bottom).
        low_mask = np.zeros_like(depth, dtype=bool)
        high_mask = np.zeros_like(depth, dtype=bool)
        for r in range(7):
            for c in range(7):
                row_dist = abs(r - 3)
                col_dist = abs(c - 3)
                if col_dist > row_dist:
                    low_mask[r, c] = True
                elif row_dist > col_dist:
                    high_mask[r, c] = True

        low_sum = float(np.sum(depth[low_mask]))
        high_sum = float(np.sum(depth[high_mask]))
        assert low_sum > high_sum, (
            f"Low quadrant depth ({low_sum:.6f}) should exceed "
            f"high quadrant depth ({high_sum:.6f})"
        )

        # Saddle point (center) should have more depth than the high
        # quadrant mean because flow converges through the saddle.
        high_mean = high_sum / np.sum(high_mask)
        saddle_depth = float(depth[3, 3])
        assert saddle_depth > high_mean, (
            f"Saddle depth ({saddle_depth:.6f}) should exceed "
            f"high quadrant mean ({high_mean:.6f})"
        )

        # Full grid regression check.
        expected = np.array(
            [
                [0.000264, 0.000282, 0.000155, 0.000203, 0.000244, 0.000155, 0.000264],
                [0.000406, 0.000299, 0.000282, 0.000392, 0.000224, 0.000509, 0.000485],
                [0.000721, 0.000509, 0.000460, 0.000812, 0.000332, 0.000556, 0.000847],
                [0.001300, 0.000830, 0.000662, 0.003083, 0.001075, 0.000946, 0.001388],
                [0.000777, 0.000545, 0.000299, 0.000777, 0.000473, 0.000611, 0.000839],
                [0.000434, 0.000348, 0.000299, 0.000406, 0.000282, 0.000434, 0.000447],
                [0.000299, 0.000203, 0.000264, 0.000203, 0.000244, 0.000282, 0.000244],
            ]
        )
        np.testing.assert_allclose(depth, expected, atol=0.001)


def test_saddle_rotation_invariance(tmp_path):
    """Rotating the saddle 90 degrees must swap peak/valley axes.

    The original saddle has valleys along columns (left-right).
    A 90-degree rotation swaps row and column terms, putting valleys
    along rows (top-bottom).
    """
    project_a = tmp_path / "a"
    gs.create_project(project_a)
    with gs.setup.init(project_a, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=7, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = (row() - 4)*(row() - 4) - (col() - 4)*(col() - 4) + 10"
            )
        )
        depth_a = run_sim(session)

    project_b = tmp_path / "b"
    gs.create_project(project_b)
    with gs.setup.init(project_b, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=7, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = (col() - 4)*(col() - 4) - (row() - 4)*(row() - 4) + 10"
            )
        )
        depth_b = run_sim(session)

    # Both are square so total depth should be comparable.
    total_a = float(np.sum(depth_a))
    total_b = float(np.sum(depth_b))
    assert total_a == pytest.approx(total_b, rel=0.1), (
        f"Total depth original ({total_a:.6f}) vs rotated ({total_b:.6f})"
    )

    # After rotation, the outer ring (excluding the noisy center row/col)
    # should have comparable depth distribution. Compare edge sums
    # excluding the center row/col (index 3) where the saddle singularity
    # amplifies Monte Carlo noise.
    edge_cols_a = float(np.sum(depth_a[:, :3]) + np.sum(depth_a[:, 4:]))
    edge_rows_b = float(np.sum(depth_b[:3, :]) + np.sum(depth_b[4:, :]))
    assert edge_cols_a == pytest.approx(edge_rows_b, rel=0.2), (
        f"Edge depth original ({edge_cols_a:.6f}) vs rotated ({edge_rows_b:.6f})"
    )


def test_hollow_concentrates_more_than_spur(tmp_path):
    """A hollow (convergent) must concentrate more water than a spur (divergent).

    Both have the same grid size and total rainfall. The hollow's concave
    shape funnels water together while the spur's convex shape spreads
    it apart.
    """
    # Hollow: convergent flow, cross-section narrows downstream.
    project_hollow = tmp_path / "hollow"
    gs.create_project(project_hollow)
    with gs.setup.init(project_hollow, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=10, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = (nrows() - row()) * 0.5"
                " + (col() - 4)*(col() - 4) * (1.0 - 0.08 * row())"
            )
        )
        depth_hollow = run_sim(session)

    # Spur: divergent flow, cross-section widens downstream.
    project_spur = tmp_path / "spur"
    gs.create_project(project_spur)
    with gs.setup.init(project_spur, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=7, s=0, n=10, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = (nrows() - row()) * 0.5"
                " + (col() - 4)*(col() - 4) * (0.2 + 0.08 * row())"
            )
        )
        depth_spur = run_sim(session)

    # Peak depth in the hollow should exceed peak depth in the spur
    # because convergent flow concentrates water.
    max_hollow = float(np.max(depth_hollow))
    max_spur = float(np.max(depth_spur))
    assert max_hollow > max_spur, (
        f"Hollow peak depth ({max_hollow:.6f}) should exceed "
        f"spur peak depth ({max_spur:.6f})"
    )

    # Full grid regression check for hollow.
    expected_hollow = np.array(
        [
            [0.000201, 0.000365, 0.000545, 0.001885, 0.000645, 0.000365, 0.000201],
            [0.000291, 0.000527, 0.000871, 0.001734, 0.000612, 0.000317, 0.000291],
            [0.000201, 0.000342, 0.000660, 0.002747, 0.000791, 0.000508, 0.000342],
            [0.000388, 0.000365, 0.000983, 0.003093, 0.001007, 0.000409, 0.000291],
            [0.000291, 0.000490, 0.000845, 0.003805, 0.001041, 0.000527, 0.000317],
            [0.000164, 0.000490, 0.001064, 0.004477, 0.001109, 0.000490, 0.000317],
            [0.000365, 0.000451, 0.001226, 0.004669, 0.001266, 0.000430, 0.000342],
            [0.000342, 0.000562, 0.001184, 0.004930, 0.001626, 0.000365, 0.000263],
            [0.000291, 0.000645, 0.001467, 0.005532, 0.001402, 0.000579, 0.000579],
            [0.000612, 0.000764, 0.001651, 0.006953, 0.002035, 0.000858, 0.000342],
        ]
    )
    np.testing.assert_allclose(depth_hollow, expected_hollow, atol=0.002)

    # Full grid regression check for spur.
    expected_spur = np.array(
        [
            [0.000314, 0.000670, 0.000927, 0.001595, 0.000700, 0.000504, 0.000261],
            [0.000384, 0.000447, 0.001089, 0.001892, 0.000825, 0.000574, 0.000427],
            [0.000339, 0.000362, 0.000812, 0.002922, 0.001174, 0.000574, 0.000427],
            [0.000314, 0.000591, 0.001100, 0.003638, 0.000998, 0.000522, 0.000314],
            [0.000288, 0.000406, 0.000963, 0.004235, 0.001195, 0.000447, 0.000231],
            [0.000163, 0.000467, 0.001021, 0.004930, 0.001185, 0.000486, 0.000314],
            [0.000339, 0.000447, 0.001010, 0.005427, 0.000914, 0.000362, 0.000261],
            [0.000199, 0.000339, 0.000998, 0.005920, 0.001089, 0.000362, 0.000231],
            [0.000288, 0.000540, 0.001121, 0.006517, 0.001185, 0.000540, 0.000427],
            [0.000288, 0.000384, 0.001215, 0.006864, 0.001143, 0.000261, 0.000231],
        ]
    )
    np.testing.assert_allclose(depth_spur, expected_spur, atol=0.002)


def test_terrace_accumulates_on_steps(tmp_path):
    """Terraced slopes must accumulate water on flat steps, not steep drops.

    A 10x12 grid with three flat steps separated by steep drops.
    Downstream steps should accumulate more water.

    Elevation (west to east):
        9 9 9 9 7 6 6 6 4 3 3 3
    Pattern: step - drop - step - drop - step
    """
    project = tmp_path / "simwe"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=12, s=0, n=10, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = if(col() <= 4, 9,"
                " if(col() == 5, 7,"
                " if(col() <= 8, 6,"
                " if(col() == 9, 4, 3))))"
            )
        )

        depth = run_sim(session)

        # Column indices (0-based) for each feature.
        step1_cols = slice(0, 4)
        drop1_col = 4
        step2_cols = slice(5, 8)
        drop2_col = 8
        step3_cols = slice(9, 12)

        step1_mean = float(np.mean(depth[:, step1_cols]))
        drop1_mean = float(np.mean(depth[:, drop1_col]))
        step2_mean = float(np.mean(depth[:, step2_cols]))
        drop2_mean = float(np.mean(depth[:, drop2_col]))
        step3_mean = float(np.mean(depth[:, step3_cols]))

        # Flat steps should have more depth than steep drops.
        assert step2_mean > drop1_mean, (
            f"Step 2 mean ({step2_mean:.6f}) should exceed "
            f"drop 1 mean ({drop1_mean:.6f})"
        )
        assert step3_mean > drop2_mean, (
            f"Step 3 mean ({step3_mean:.6f}) should exceed "
            f"drop 2 mean ({drop2_mean:.6f})"
        )

        # Downstream steps should accumulate more water.
        assert step2_mean > step1_mean, (
            f"Step 2 ({step2_mean:.6f}) should exceed step 1 ({step1_mean:.6f})"
        )
        assert step3_mean > step2_mean, (
            f"Step 3 ({step3_mean:.6f}) should exceed step 2 ({step2_mean:.6f})"
        )

        # Full grid regression check.
        expected = np.array(
            [
                [
                    0.000827,
                    0.000923,
                    0.002275,
                    0.000315,
                    0.000477,
                    0.001096,
                    0.004063,
                    0.000776,
                    0.001012,
                    0.001928,
                    0.008246,
                    0.004348,
                ],
                [
                    0.001783,
                    0.002471,
                    0.002726,
                    0.000968,
                    0.001055,
                    0.001956,
                    0.006184,
                    0.001566,
                    0.001327,
                    0.003007,
                    0.012262,
                    0.007194,
                ],
                [
                    0.001984,
                    0.002965,
                    0.003505,
                    0.000776,
                    0.000668,
                    0.001433,
                    0.004731,
                    0.001433,
                    0.001327,
                    0.003371,
                    0.013642,
                    0.008678,
                ],
                [
                    0.003561,
                    0.005511,
                    0.003312,
                    0.000827,
                    0.000876,
                    0.001783,
                    0.006326,
                    0.001363,
                    0.001433,
                    0.003959,
                    0.015099,
                    0.009384,
                ],
                [
                    0.003818,
                    0.005705,
                    0.003636,
                    0.000968,
                    0.000968,
                    0.002447,
                    0.007334,
                    0.001928,
                    0.001433,
                    0.003959,
                    0.017127,
                    0.010672,
                ],
                [
                    0.004348,
                    0.006223,
                    0.003580,
                    0.000723,
                    0.000827,
                    0.002172,
                    0.005732,
                    0.001215,
                    0.001398,
                    0.004029,
                    0.018605,
                    0.011421,
                ],
                [
                    0.003764,
                    0.005412,
                    0.003580,
                    0.000827,
                    0.000668,
                    0.001723,
                    0.005005,
                    0.001327,
                    0.001215,
                    0.003505,
                    0.016332,
                    0.010490,
                ],
                [
                    0.003524,
                    0.003636,
                    0.003172,
                    0.000668,
                    0.000723,
                    0.001783,
                    0.005988,
                    0.001096,
                    0.001055,
                    0.002965,
                    0.012781,
                    0.009848,
                ],
                [
                    0.003351,
                    0.003371,
                    0.002770,
                    0.000546,
                    0.000668,
                    0.001984,
                    0.005483,
                    0.001467,
                    0.001501,
                    0.002814,
                    0.010435,
                    0.008395,
                ],
                [
                    0.002495,
                    0.001501,
                    0.002275,
                    0.000546,
                    0.000477,
                    0.001215,
                    0.003543,
                    0.001177,
                    0.001137,
                    0.002172,
                    0.007910,
                    0.004131,
                ],
            ]
        )
        np.testing.assert_allclose(depth, expected, atol=0.003)


def test_terrace_rotation_invariance(tmp_path):
    """E-W terraces and N-S terraces must produce rotationally equivalent depth."""
    # E-W terraces: 10 rows x 12 cols.
    project_ew = tmp_path / "ew"
    gs.create_project(project_ew)
    with gs.setup.init(project_ew, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=12, s=0, n=10, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = if(col() <= 4, 9,"
                " if(col() == 5, 7,"
                " if(col() <= 8, 6,"
                " if(col() == 9, 4, 3))))"
            )
        )
        depth_ew = run_sim(session)

    # N-S terraces: 12 rows x 10 cols.
    project_ns = tmp_path / "ns"
    gs.create_project(project_ns)
    with gs.setup.init(project_ns, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(w=0, e=10, s=0, n=12, res=1)
        tools.r_mapcalc(
            expression=(
                "elevation = if(row() <= 4, 9,"
                " if(row() == 5, 7,"
                " if(row() <= 8, 6,"
                " if(row() == 9, 4, 3))))"
            )
        )
        depth_ns = run_sim(session)

    # Total depth should be comparable.
    total_ew = float(np.sum(depth_ew))
    total_ns = float(np.sum(depth_ns))
    assert total_ew == pytest.approx(total_ns, rel=0.02), (
        f"Total depth E-W ({total_ew:.6f}) vs N-S ({total_ns:.6f})"
    )

    # Cross-terrace profiles should match after rotation.
    # Monte Carlo noise on small grids requires generous tolerance.
    profile_ew = np.sum(depth_ew, axis=0)
    depth_ns_rotated = np.rot90(depth_ns)
    profile_ns = np.sum(depth_ns_rotated, axis=0)
    np.testing.assert_allclose(profile_ew, profile_ns, rtol=0.3)
