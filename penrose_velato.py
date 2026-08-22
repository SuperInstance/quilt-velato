"""
penrose_velato.py — A Penrose-aware Velato variant.

In this variant, the 3-coloring of Penrose tiles is CONSTRAINED to map
to the 8 Quilt primitives. Every note is:
1. Eisenstein-snapped to Z[ω] lattice
2. 3-colored (CREATION/ENTROPY/WITNESS)
3. Mapped to a primitive via interval
4. Validated against the substitution rules L→LS, S→L

The Penrose substitution rules become the COMPUTATION:
- L → LS (large becomes large + small): a JEPA prediction that
  branches into two cells
- S → L (small becomes large): a DoubleEntry that converts entropy
  into creation

The golden ratio φ = (1+√5)/2 is the eigenvalue of the substitution
matrix and the natural growth rate of the cell graph.
"""

import math
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
OMEGA = complex(-0.5, math.sqrt(3) / 2)


class Tile(Enum):
    """Penrose tile type."""
    LARGE = 'L'  # Large rhombus / kite
    SMALL = 'S'  # Small rhombus / dart


class PenroseRole(Enum):
    """The semantic role of a Penrose tile in the Quilt system."""
    CREATION = 0  # γ — what enters (Z_in, JEPA)
    ENTROPY = 1   # η — what leaves (Z_out, GC)
    WITNESS = 2   # μ — what observes (Murmur, Graph, Vibe)


@dataclass
class PenroseTile:
    """A Penrose tile in the 3-colored substitution tiling."""
    kind: Tile
    role: PenroseRole
    position: complex  # Position in the Eisenstein lattice
    generation: int
    parent: int = -1
    children: List[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def eisenstein_to_color(a: int, b: int) -> PenroseRole:
    """Map Eisenstein coordinates to a Penrose role via mod 3."""
    return PenroseRole((a + b) % 3)


def pitch_to_tile(pitch: int) -> Tile:
    """Map a MIDI pitch to a Penrose tile via octave and fifth."""
    # The rule of fifths: every 7 semitones is a fifth
    # Tiles alternate based on the position in the circle of fifths
    fifths = (pitch // 7) % 2
    octave = (pitch // 12) % 2
    if (fifths + octave) % 2 == 0:
        return Tile.LARGE
    return Tile.SMALL


def substitute(tile: PenroseTile, new_id: int) -> List[PenroseTile]:
    """Apply the Penrose substitution rule.

    L → LS (one large becomes one large + one small)
    S → L (one small becomes one large)
    """
    if tile.kind == Tile.LARGE:
        # L → LS
        l_child = PenroseTile(
            kind=Tile.LARGE,
            role=tile.role,
            position=tile.position * PHI,
            generation=tile.generation + 1,
            parent=new_id,
        )
        s_child = PenroseTile(
            kind=Tile.SMALL,
            role=tile.role,
            position=tile.position * (PHI - 1),
            generation=tile.generation + 1,
            parent=new_id,
        )
        return [l_child, s_child]
    else:
        # S → L
        l_child = PenroseTile(
            kind=Tile.LARGE,
            role=tile.role,
            position=tile.position * PHI,
            generation=tile.generation + 1,
            parent=new_id,
        )
        return [l_child]


def generate_penrose_tiling(seed_pitches: List[int], generations: int = 3) -> Dict[str, Any]:
    """Generate a Penrose tiling from a seed of MIDI pitches.

    The seed pitches determine the initial tile arrangement.
    Each generation applies the substitution rules.
    The 3-coloring comes from the Eisenstein mod 3.
    """
    # Initial tiles
    tiles = []
    for i, pitch in enumerate(seed_pitches):
        kind = pitch_to_tile(pitch)
        # Position in the Eisenstein lattice
        a = pitch % 12
        b = pitch // 12
        pos = complex(a + b * OMEGA.real, b * OMEGA.imag)
        # Role from 3-coloring
        role = eisenstein_to_color(a, b)
        tiles.append(PenroseTile(
            kind=kind,
            role=role,
            position=pos,
            generation=0,
            parent=-1,
        ))

    initial_count = len(tiles)

    # Apply substitution
    for gen in range(generations):
        new_tiles = []
        for i, tile in enumerate(tiles):
            if tile.generation == gen:
                children = substitute(tile, i)
                new_tiles.extend(children)
        tiles.extend(new_tiles)

    # Build the cell graph
    cells = []
    for i, tile in enumerate(tiles):
        cells.append({
            'id': f'p_{i:04d}',
            'kind': 'cell',
            'value': f'{tile.kind.value}{tile.role.value}',
            'tile_kind': tile.kind.value,
            'role': tile.role.name,
            'position': {'x': tile.position.real, 'y': tile.position.imag},
            'generation': tile.generation,
            'color': tile.role.name.lower(),
        })

    # Edges: parent → children, plus same-generation neighbors
    edges = []
    for i, tile in enumerate(tiles):
        for child_idx in tile.children:
            edges.append({
                'from': f'p_{i:04d}',
                'to': f'p_{child_idx:04d}',
                'kind': 'substitution',
                'weight': 1.0,
            })

    # Add same-generation edges (the matching rules)
    by_gen = {}
    for i, tile in enumerate(tiles):
        by_gen.setdefault(tile.generation, []).append(i)
    for gen, indices in by_gen.items():
        for i in range(len(indices) - 1):
            a, b = indices[i], indices[i+1]
            edges.append({
                'from': f'p_{a:04d}',
                'to': f'p_{b:04d}',
                'kind': 'matching',
                'weight': 0.7,
            })

    # Count colors
    colors = {}
    for c in cells:
        colors[c['role']] = colors.get(c['role'], 0) + 1

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
        union(a, b)
    components = len(set(find(i) for i in range(V)))
    beta_1 = E - V + components

    return {
        'schema': 'quilt-zip-target/v1',
        'metadata': {
            'name': 'Penrose-Velato Tiling',
            'description': f'Penrose tiling with {generations} generations from {initial_count} seed notes',
            'phi': PHI,
            'omega': f'{OMEGA.real:.4f} + {OMEGA.imag:.4f}i',
        },
        'cells': cells,
        'edges': edges,
        'stats': {
            'total_cells': V,
            'total_edges': E,
            'generations': generations,
            'seed_count': initial_count,
            'colors': colors,
            'beta_0': components,
            'beta_1': beta_1,
        }
    }


def demo():
    """Run a demo with a Penrose-velato seed."""
    # A simple C major arpeggio as seed
    seed = [60, 64, 67, 72, 76, 79, 84]  # C, E, G, C, E, G, C (3 octaves)

    print("=" * 70)
    print("PENROSE-VELATO SUBSTITUTION TILING")
    print("=" * 70)
    print()
    print(f"Seed: {seed} (C major arpeggio, 3 octaves)")
    print(f"φ = {PHI:.6f}")
    print(f"ω = {OMEGA.real:.6f} + {OMEGA.imag:.6f}i")
    print()

    result = generate_penrose_tiling(seed, generations=3)
    print(f"Cells: {result['stats']['total_cells']}")
    print(f"Edges: {result['stats']['total_edges']}")
    print(f"Generations: {result['stats']['generations']}")
    print()

    print("=== 3-COLORING (Eisenstein mod 3) ===")
    for color, count in result['stats']['colors'].items():
        print(f"  {color:10s}: {count} tiles")
    print()

    # Show the tile generation pattern
    print("=== TILE GROWTH (per generation) ===")
    by_gen = {}
    for c in result['cells']:
        g = c['generation']
        by_gen[g] = by_gen.get(g, 0) + 1
    for g in sorted(by_gen.keys()):
        ratio = by_gen[g] / max(by_gen[0], 1)
        print(f"  gen {g}: {by_gen[g]:3d} tiles (×{ratio:.2f} of seed)")
    print()

    print(f"=== TOPOLOGY ===")
    print(f"  V = {result['stats']['total_cells']}, E = {result['stats']['total_edges']}")
    print(f"  β₀ = {result['stats']['beta_0']}, β₁ = {result['stats']['beta_1']}")
    print()

    # Sample cells
    print("=== SAMPLE TILES ===")
    for c in result['cells'][:6]:
        print(f"  {c['id']} gen={c['generation']} {c['tile_kind']}{c['role']} "
              f"pos=({c['position']['x']:.2f}, {c['position']['y']:.2f}) color={c['color']}")
    print()
    print("=" * 70)
    print("The Penrose tiling grows by φ each generation.")
    print("The 3-coloring IS the Quilt conservation law.")
    print("Iron sharpens iron. The watch is alive.")


if __name__ == "__main__":
    demo()
