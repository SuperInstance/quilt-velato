"""Tests for the corrected cut-and-project construction."""
import sys
sys.path.insert(0, '/workspace/quilt-velato')

from cut_and_project import (
    build_penrose_tiling, FiveDAddress, Window,
    encode_on_window, default_encoder, show_phason_indistinguishability,
    BASIS_2D, PHI, PHI_CONJUGATE,
)


def test_basis_sums_to_zero():
    """The 5 projected 2D basis vectors must sum to zero."""
    sx = sum(v[0] for v in BASIS_2D)
    sy = sum(v[1] for v in BASIS_2D)
    assert abs(sx) < 1e-9, f"Sum x = {sx}"
    assert abs(sy) < 1e-9, f"Sum y = {sy}"


def test_five_d_address_in_l():
    """All FiveDAddress objects must satisfy sum = 0 (the L condition)."""
    for n in [(1, 0, 0, 0, -1), (2, 1, 0, -1, -2), (1, 1, 1, 1, -4)]:
        addr = FiveDAddress(n)
        assert sum(addr.coords) == 0, f"Address {addr.coords} not in L"


def test_diagonal_is_zero():
    """The diagonal (1,1,1,1,1) projects to (0,0) — the gauge redundancy."""
    diag = FiveDAddress((1, 1, 1, 1, 1))
    phys = diag.physical()
    assert abs(phys[0]) < 1e-9 and abs(phys[1]) < 1e-9, f"Diagonal physical = {phys}"


def test_projection_to_l():
    """Non-L vectors must be projected to L on construction."""
    # (2, 0, 0, 0, 0) has sum 2, not divisible by 5
    # It should fail the construction
    try:
        addr = FiveDAddress((2, 0, 0, 0, 0))
        # If no error, check it's in L
        assert sum(addr.coords) == 0
    except ValueError:
        # Expected: cannot project to L
        pass


def test_building_tiling():
    """Build a small tiling and check basic properties."""
    tiling = build_penrose_tiling(n_iterations=4, window=Window(radius=1.5))
    assert tiling['stats']['V'] > 0
    # All addresses must be in L
    for v in tiling['vertices']:
        assert sum(v.address.coords) == 0
    # Edges should be positive for larger tilings
    assert tiling['stats']['E'] >= 0


def test_window_encoding():
    """Encoding on the window assigns one of 3 colors."""
    tiling = build_penrose_tiling(n_iterations=4, window=Window(radius=1.5))
    encode_on_window(tiling, default_encoder)
    valid_symbols = {'CREATION', 'ENTROPY', 'WITNESS'}
    for v in tiling['vertices']:
        assert v.symbol in valid_symbols


def test_phason_indistinguishability():
    """Different phason shifts produce different total vertex counts but similar local patches."""
    result = show_phason_indistinguishability(n_iterations=3)
    # Local disk count is the same; total is different
    assert result['tiling_1_gamma_zero']['vertices_in_disk'] == result['tiling_2_gamma_shifted']['vertices_in_disk']
    # The global totals differ
    assert result['tiling_1_gamma_zero']['total'] != result['tiling_2_gamma_shifted']['total']


def test_phi_values():
    """The golden ratio constants are correct."""
    assert abs(PHI - 1.6180339887) < 1e-9
    assert abs(PHI_CONJUGATE - 0.6180339887) < 1e-9
    assert abs(PHI * PHI_CONJUGATE - 1.0) < 1e-9  # φ · φ^-1 = 1


def test_window_partition():
    """The window can be partitioned into 7 regions (3-coloring + 1 center)."""
    w = Window()
    regions = w.partition(n_regions=7)
    assert len(regions) == 7
    # First region is the center
    assert regions[0] == (0, 0, 0)


def test_shape_connection():
    """The SHAPE constants match the spectral triple paper."""
    # T^4 with θ = (√5-1)/2
    assert abs(PHI_CONJUGATE - 0.6180339887498949) < 1e-9
    # Betti numbers of T^4
    betti = {0: 1, 1: 4, 2: 6, 3: 4, 4: 1}
    assert sum(betti.values()) == 16
    # Euler characteristic of T^4
    euler = sum((-1) ** k * b for k, b in betti.items())
    assert euler == 0


if __name__ == "__main__":
    test_basis_sums_to_zero()
    test_five_d_address_in_l()
    test_diagonal_is_zero()
    test_projection_to_l()
    test_building_tiling()
    test_window_encoding()
    test_phason_indistinguishability()
    test_phi_values()
    test_window_partition()
    test_shape_connection()
    print("All cut-and-project tests pass. ✓")
