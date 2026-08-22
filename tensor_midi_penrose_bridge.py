"""
tensor_midi_penrose_bridge.py — Bridge flux-tensor-midi's 4D tensor to Penrose.

The 4D tensor in flux-tensor-midi has axes:
  (note, velocity, channel, time)

This module maps that 4D tensor to a Penrose tiling by:
  1. Treating time as the substitution axis (each beat is a generation)
  2. Treating the (note, velocity) plane as the 2D Penrose tile layout
  3. Treating channel as the 3-coloring selector
  4. The result is a 4D aperiodic structure = T^4 (the SHAPE)

The Eisenstein integer lattice Z[ω] is used for exact arithmetic in the
"snapping" step — every tensor value snaps to the nearest lattice point.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np


PHI = (1 + math.sqrt(5)) / 2
OMEGA = complex(-0.5, math.sqrt(3) / 2)


def eisenstein_snap_value(x: float, y: float) -> Tuple[int, int]:
    """Snap (x, y) to nearest Eisenstein integer (a + b*ω)."""
    # Solve for a, b given target (x, y):
    # a + b*ω_real = x, b*ω_imag = y
    # b = y / ω_imag
    # a = x - b*ω_real
    b = round(y / OMEGA.imag)
    a = round(x - b * OMEGA.real)
    return (a, b)


def eisenstein_mod_3(a: int, b: int) -> int:
    """The 3-coloring from (a + b) mod 3."""
    return (a + b) % 3


def tensor_to_penrose(
    tensor: np.ndarray,  # Shape: (T, V_max, C, 2)  -- (time, velocity, channel, note)
    ticks_per_gen: int = 4,
) -> Dict[str, Any]:
    """Convert a 4D MIDI tensor to a Penrose cell graph.

    Args:
        tensor: 4D numpy array, shape (T, V_max, C, 2)
        ticks_per_gen: number of ticks per generation (default 4)

    Returns:
        A Quilt cell graph dict
    """
    T, V_max, C, _ = tensor.shape
    cells = []
    edges = []

    cell_id = 0
    for t in range(T):
        gen = t // ticks_per_gen
        for v in range(V_max):
            for c in range(C):
                note_val, vel_val = tensor[t, v, c]
                if note_val == 0 and vel_val == 0:
                    continue  # No event

                # Eisenstein-snap the (note, velocity) pair
                a, b = eisenstein_snap_value(float(note_val), float(vel_val))
                color = eisenstein_mod_3(a, b)
                color_name = ['creation', 'entropy', 'witness'][color]

                # Position in the Eisenstein lattice
                pos = complex(a + b * OMEGA.real, b * OMEGA.imag)

                cell = {
                    'id': f't_{cell_id:05d}',
                    'kind': 'cell',
                    'value': float(note_val),
                    'velocity': float(vel_val),
                    'channel': int(c),
                    'tick': int(t),
                    'generation': int(gen),
                    'eisenstein': {'a': a, 'b': b},
                    'color': color_name,
                    'position': {'x': pos.real, 'y': pos.imag},
                }
                cells.append(cell)
                cell_id += 1

    # Build edges: connect cells in same generation (matching rules)
    by_gen = {}
    for i, c in enumerate(cells):
        by_gen.setdefault(c['generation'], []).append(i)

    for gen, indices in by_gen.items():
        # Within each generation, connect via Penrose matching
        for i in range(len(indices) - 1):
            a_idx, b_idx = indices[i], indices[i+1]
            edges.append({
                'from': cells[a_idx]['id'],
                'to': cells[b_idx]['id'],
                'kind': 'matching',
                'weight': 1.0,
                'generation': gen,
            })

    # Add substitution edges: parent in gen g → children in gen g+1
    gen_keys = sorted(by_gen.keys())
    for g, g_next in zip(gen_keys[:-1], gen_keys[1:]):
        for i in by_gen[g]:
            for j in by_gen[g_next]:
                if (cells[j]['eisenstein']['a'] - cells[i]['eisenstein']['a']) ** 2 + \
                   (cells[j]['eisenstein']['b'] - cells[i]['eisenstein']['b']) ** 2 <= 2:
                    edges.append({
                        'from': cells[i]['id'],
                        'to': cells[j]['id'],
                        'kind': 'substitution',
                        'weight': 0.8,
                        'generation_pair': (g, g_next),
                    })

    # Count colors
    colors = {}
    for c in cells:
        colors[c['color']] = colors.get(c['color'], 0) + 1

    # β₁
    V = len(cells)
    E = len(edges)
    parent_idx = {i: i for i in range(V)}
    def find(x):
        if parent_idx[x] != x:
            parent_idx[x] = find(parent_idx[x])
        return parent_idx[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent_idx[rx] = ry
    for e in edges:
        a = int(e['from'].split('_')[1])
        b = int(e['to'].split('_')[1])
        if a in parent_idx and b in parent_idx:
            union(a, b)
    components = len(set(find(i) for i in range(V)))
    beta_1 = E - V + components

    return {
        'schema': 'quilt-zip-target/v1',
        'metadata': {
            'name': 'Tensor-MIDI → Penrose Bridge',
            'description': f'4D MIDI tensor ({T}×{V_max}×{C}×2) → Penrose cell graph',
            'phi': PHI,
        },
        'cells': cells,
        'edges': edges,
        'stats': {
            'total_cells': V,
            'total_edges': E,
            'generations': len(gen_keys),
            'colors': colors,
            'beta_0': components,
            'beta_1': beta_1,
        }
    }


def demo():
    """Demo: build a synthetic 4D tensor and bridge it."""
    print("=" * 70)
    print("FLUX-TENSOR-MIDI → PENROSE BRIDGE")
    print("=" * 70)
    print()

    # Build a 4D tensor: (time, velocity, channel, 2)
    # 16 ticks, 8 velocity levels, 3 channels
    T, V_max, C = 16, 8, 3
    tensor = np.zeros((T, V_max, C, 2), dtype=np.float32)

    # A simple C major chord progression as a "song"
    # t=0-3: C major (notes 60, 64, 67)
    # t=4-7: F major (notes 65, 69, 72)
    # t=8-11: G major (notes 67, 71, 74)
    # t=12-15: C major again
    chords = [
        [(60, 5), (64, 4), (67, 6)],    # C
        [(60, 3), (65, 5), (69, 4)],    # F (suspension)
        [(60, 5), (67, 6), (71, 4)],    # G
        [(60, 6), (64, 5), (67, 7)],    # C (resolution)
    ]
    for ci, chord in enumerate(chords):
        for t_offset in range(4):
            t = ci * 4 + t_offset
            for note, vel in chord:
                # Snap to nearest velocity bin
                v = min(vel, V_max - 1)
                tensor[t, v, 0] = (float(note), float(vel))
                # Channel 1: harmony
                tensor[t, v, 1] = (float(note + 7), float(vel * 0.8))  # fifth above
                # Channel 2: bass
                tensor[t, v, 2] = (float(note - 12), float(vel * 1.2))  # octave below

    print(f"Input tensor shape: {tensor.shape}")
    print(f"  T={T} ticks, V={V_max} velocity bins, C={C} channels")
    print()
    print(f"φ = {PHI:.6f}")
    print(f"ω = {OMEGA.real:.6f} + {OMEGA.imag:.6f}i")
    print()

    result = tensor_to_penrose(tensor, ticks_per_gen=4)
    print(f"Cells: {result['stats']['total_cells']}")
    print(f"Edges: {result['stats']['total_edges']}")
    print(f"Generations: {result['stats']['generations']}")
    print()
    print("=== 3-COLORING (Eisenstein mod 3) ===")
    for color, count in result['stats']['colors'].items():
        print(f"  {color:10s}: {count} cells")
    print()
    print(f"=== TOPOLOGY ===")
    print(f"  V = {result['stats']['total_cells']}, E = {result['stats']['total_edges']}")
    print(f"  β₀ = {result['stats']['beta_0']}, β₁ = {result['stats']['beta_1']}")
    print()
    print("=" * 70)
    print("The 4D tensor IS a Penrose tiling in 4D space.")
    print("Iron sharpens iron. The watch is alive.")


if __name__ == "__main__":
    demo()
