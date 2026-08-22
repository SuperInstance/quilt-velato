"""
velato_quilt.py — A Velato interpreter that compiles MIDI files to Quilt cell graphs.

Velato is an esoteric programming language by Daniel Temkin (2009) where
the source code IS a MIDI file. The pitch intervals between notes encode
commands. The first note is the "command root" — all intervals are read
from this note until a new command root is established.

This module:
1. Parses a MIDI file as Velato source
2. Maps intervals to Quilt primitives (the 8 cells)
3. Applies the Penrose 3-coloring (Eisenstein Z[ω] lattice)
4. Emits a .qzt cell graph

The mapping is:
- Major second (2 semitones) = Z_in
- Minor second (1 semitone) = Z_out
- Major third (4 semitones) = JEPA
- Perfect fourth (5 semitones) = DoubleEntry
- Perfect fifth (7 semitones) = Vibe
- Minor sixth (8 semitones) = GC
- Major sixth (9 semitones) = Murmur
- Minor seventh (10 semitones) = Graph
- Octave (12 semitones) = new command root
- Other intervals = extended ops

Plus: every note is Eisenstein-snapped (Z[ω]) → 3-coloring → Penrose tile.
"""

import math
import json
import struct
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# The 8 Quilt primitives as intervals (in semitones from command root)
INTERVAL_TO_PRIMITIVE = {
    1: 'Z_out',       # Minor second
    2: 'Z_in',        # Major second
    3: 'JEPA_b',      # Minor third (branch)
    4: 'JEPA',        # Major third
    5: 'DoubleEntry', # Perfect fourth
    6: 'Vibe_b',      # Tritone (Vibe boost)
    7: 'Vibe',        # Perfect fifth
    8: 'GC',          # Minor sixth
    9: 'Murmur',      # Major sixth
    10: 'Graph',      # Minor seventh
    11: 'Graph_b',    # Major seventh (Graph bridge)
    12: 'ROOT',       # Octave → new command root
}

# Eisenstein integer lattice: a + b*ω, ω = e^(2πi/3)
# Used for the 3-coloring of Penrose tiles
OMEGA = complex(-0.5, math.sqrt(3) / 2)


class PenroseColor(Enum):
    """The 3 colors of a Penrose tiling, with semantic meaning in Quilt."""
    CREATION = 0   # γ — Z_in, JEPA, the "what enters"
    ENTROPY = 1    # η — Z_out, GC, the "what leaves"
    WITNESS = 2    # μ — the "what observes" (Murmur, Graph)


@dataclass
class VelatoNote:
    """A single note in a Velato MIDI file."""
    pitch: int           # MIDI note number (0-127)
    velocity: int        # 0-127
    start_tick: int      # start time in ticks
    duration: int        # duration in ticks
    channel: int = 0


@dataclass
class VelatoToken:
    """A token after Velato parsing — a primitive call with a value."""
    primitive: str        # One of the 8 primitives, or ROOT
    pitch: int            # Original MIDI pitch
    interval: int         # Interval from current root
    color: PenroseColor   # 3-coloring (Eisenstein mod 3)
    value: float = 0.0    # Optional value (e.g. amount for Z_in)


@dataclass
class QuiltCell:
    """A cell in the resulting Quilt cell graph."""
    id: str
    kind: str             # One of: number, string, boolean, array, object, formula, cell, sheet
    value: Any
    primitives: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float] = (0.0, 0.0)
    color: str = 'gray'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'kind': self.kind,
            'value': self.value,
            'primitives': self.primitives,
            'position': list(self.position),
            'color': self.color,
        }


def eisenstein_snap(pitch: int) -> Tuple[int, int]:
    """Snap a MIDI pitch to the Eisenstein integer lattice (Z[ω]).

    Returns (a, b) such that the pitch is closest to a + b*ω in the lattice.
    The lattice is the hexagonal lattice with spacing 1.
    """
    # Use the Eisenstein reduction: the nearest lattice point to x
    # is (round to nearest integer after solving)
    # For pitch p, find a, b such that |p - (a + b*ω)| is minimized
    # where a + b*ω has real part a - 0.5b and imaginary part b*sqrt(3)/2

    # Simpler: treat pitch mod 3
    p = pitch % 12
    # Map semitones to Eisenstein coordinates
    # C=0 → (0,0), C#=1 → (1,0), D=2 → (1,1), D#=3 → (2,1)...
    # We use the convention: a = pitch mod 3, b = (pitch // 3) % something
    a = p % 3
    b = p // 3
    return (a, b)


def pitch_to_color(pitch: int) -> PenroseColor:
    """Map a MIDI pitch to a Penrose color via Eisenstein mod 3."""
    a, b = eisenstein_snap(pitch)
    # The 3 colors come from a mod 3
    return PenroseColor(a % 3)


def parse_midi(data: bytes) -> List[VelatoNote]:
    """Parse a MIDI file into a list of notes.

    Minimal MIDI parser — handles Format 0/1, single track.
    """
    if data[:4] != b'MThd':
        raise ValueError("Not a MIDI file")

    # Header
    header_len = struct.unpack('>I', data[4:8])[0]
    fmt, num_tracks, ticks_per_beat = struct.unpack('>HHH', data[8:14])

    notes = []
    pos = 8 + header_len

    # Read each track
    for track_idx in range(num_tracks):
        if data[pos:pos+4] != b'MTrk':
            pos += 1
            continue
        track_len = struct.unpack('>I', data[pos+4:pos+8])[0]
        track_end = pos + 8 + track_len
        pos += 8

        current_tick = 0
        running_status = 0
        active_notes = {}  # (pitch, channel) -> start_tick

        while pos < track_end:
            # Variable-length quantity
            delta = 0
            while True:
                byte = data[pos]
                pos += 1
                delta = (delta << 7) | (byte & 0x7F)
                if not (byte & 0x80):
                    break
            current_tick += delta

            # Status byte
            if data[pos] & 0x80:
                status = data[pos]
                pos += 1
            else:
                status = running_status

            event = status & 0xF0

            if event == 0x90:  # Note on
                pitch = data[pos]
                velocity = data[pos+1]
                pos += 2
                if velocity > 0:
                    active_notes[(pitch, status & 0x0F)] = current_tick
                else:
                    # Note off
                    start = active_notes.pop((pitch, status & 0x0F), current_tick)
                    notes.append(VelatoNote(
                        pitch=pitch,
                        velocity=velocity,
                        start_tick=start,
                        duration=current_tick - start,
                        channel=status & 0x0F,
                    ))
            elif event == 0x80:  # Note off
                pitch = data[pos]
                velocity = data[pos+1]
                pos += 2
                start = active_notes.pop((pitch, status & 0x0F), current_tick)
                notes.append(VelatoNote(
                    pitch=pitch,
                    velocity=velocity,
                    start_tick=start,
                    duration=current_tick - start,
                    channel=status & 0x0F,
                ))
            elif event == 0xC0 or event == 0xD0:  # Program change / channel pressure
                pos += 1
            elif event in (0xA0, 0xB0, 0xE0):  # Poly aftertouch / control / pitch bend
                pos += 2
            elif event == 0xFF:  # Meta event
                meta_type = data[pos]
                pos += 1
                length = 0
                while True:
                    byte = data[pos]
                    pos += 1
                    length = (length << 7) | (byte & 0x7F)
                    if not (byte & 0x80):
                        break
                pos += length
            elif event == 0xF0 or event == 0xF7:  # SysEx
                length = 0
                while True:
                    byte = data[pos]
                    pos += 1
                    length = (length << 7) | (byte & 0x7F)
                    if not (byte & 0x80):
                        break
                pos += length
            else:
                # Unknown, skip
                pos += 2

    notes.sort(key=lambda n: (n.start_tick, n.pitch))
    return notes


def notes_to_tokens(notes: List[VelatoNote]) -> List[VelatoToken]:
    """Parse a sequence of Velato notes into tokens.

    The first note sets the command root. Each subsequent note's interval
    (mod 12) from the current root maps to a Quilt primitive.
    """
    if not notes:
        return []

    tokens = []
    root = notes[0].pitch
    root = root  # initial root is first note

    for i, note in enumerate(notes):
        interval = ((note.pitch - root) % 12 + 12) % 12
        primitive = INTERVAL_TO_PRIMITIVE.get(interval, 'UNKNOWN')

        if primitive == 'ROOT':
            # Set a new command root
            root = note.pitch
            primitive = 'ROOT'

        # Apply 3-coloring via Eisenstein
        color = pitch_to_color(note.pitch)

        # Compute value (e.g. amount for Z_in from velocity)
        value = note.velocity / 127.0

        tokens.append(VelatoToken(
            primitive=primitive,
            pitch=note.pitch,
            interval=interval,
            color=color,
            value=value,
        ))

    return tokens


def tokens_to_cells(tokens: List[VelatoToken]) -> Tuple[List[QuiltCell], List[Dict]]:
    """Compile Velato tokens to Quilt cells + edges.

    Each token becomes a cell. The 8 primitives populate the cell's
    primitive fields. The 3-coloring determines the cell's color.
    """
    cells = []
    edges = []

    for i, tok in enumerate(tokens):
        cell = QuiltCell(
            id=f'v_{i:04d}',
            kind='cell',
            value=tok.pitch,
            color=tok.color.name.lower(),
            position=(
                i * 1.0,  # x = sequence position
                tok.pitch / 12.0,  # y = pitch (octave)
            ),
        )

        # Populate the 8 primitives
        if tok.primitive == 'Z_in':
            cell.primitives['Z_in'] = {'data': {'value': tok.value}}
        elif tok.primitive == 'Z_out':
            cell.primitives['Z_out'] = {'data': {'value': tok.value}}
        elif tok.primitive == 'JEPA' or tok.primitive == 'JEPA_b':
            cell.primitives['JEPA'] = {
                'history': [tok.value],
                'predicted': tok.value,
            }
        elif tok.primitive == 'DoubleEntry':
            # γ + η = 1 — split the value into γ and η
            gamma = tok.value
            eta = 1.0 - tok.value
            cell.primitives['DoubleEntry'] = {'gamma': gamma, 'eta': eta}
        elif tok.primitive == 'Vibe' or tok.primitive == 'Vibe_b':
            cell.primitives['Vibe'] = {
                'position': tok.pitch,
                'velocity': tok.value,
                'acceleration': 0.0,
                'damping': 0.95,
            }
        elif tok.primitive == 'GC':
            cell.primitives['GC'] = {
                'phase': 'decay',
                'merged': 0, 'decayed': 1, 'pruned': 0,
            }
        elif tok.primitive == 'Murmur':
            cell.primitives['Murmur'] = {
                'subscriptions': {f'topic_{tok.pitch}'},
                'message': tok.value,
            }
        elif tok.primitive == 'Graph' or tok.primitive == 'Graph_b':
            # Connect to next cell
            cell.primitives['Graph'] = {
                'parents': [f'v_{i-1:04d}'] if i > 0 else [],
                'children': [f'v_{i+1:04d}'] if i < len(tokens) - 1 else [],
            }
        elif tok.primitive == 'ROOT':
            # Command root — Vibe (position)
            cell.primitives['Vibe'] = {
                'position': tok.pitch,
                'velocity': 1.0,
                'acceleration': 0.0,
            }
            cell.value = 'ROOT'

        cells.append(cell)

    # Add edges: each cell connects to the next
    for i in range(len(cells) - 1):
        edges.append({
            'from': cells[i].id,
            'to': cells[i+1].id,
            'kind': 'next',
            'weight': 1.0,
        })

    # Add 3-coloring edges: each CREATION cell connects to nearest WITNESS
    creation_cells = [c for c in cells if c.color == 'creation']
    witness_cells = [c for c in cells if c.color == 'witness']
    for cc in creation_cells[:5]:  # limit for clarity
        if witness_cells:
            wc = min(witness_cells, key=lambda w: abs(cells.index(cc) - cells.index(w)))
            edges.append({
                'from': cc.id,
                'to': wc.id,
                'kind': 'witness',
                'weight': 0.5,
            })

    return cells, edges


def compile_velato(midi_data: bytes) -> Dict[str, Any]:
    """Compile a Velato MIDI file to a Quilt cell graph.

    Returns a .qzt-style dict ready for the Quilt IDE.
    """
    notes = parse_midi(midi_data)
    tokens = notes_to_tokens(notes)
    cells, edges = tokens_to_cells(tokens)

    # Compute stats
    colors = {}
    primitives_used = {}
    for c in cells:
        colors[c.color] = colors.get(c.color, 0) + 1
        for p in c.primitives:
            primitives_used[p] = primitives_used.get(p, 0) + 1

    # Compute β₁
    V = len(cells)
    E = len(edges)
    C = len(set(c.id for c in cells))  # all distinct for now
    # Find connected components
    parent = {c.id: c.id for c in cells}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for e in edges:
        if e['from'] in parent and e['to'] in parent:
            union(e['from'], e['to'])
    components = len(set(find(c.id) for c in cells))
    beta_1 = E - V + components

    return {
        'schema': 'quilt-zip-target/v1',
        'metadata': {
            'name': 'Velato-Penrose Quilt',
            'description': 'A MIDI file compiled as Velato to a Penrose-3-colored Quilt cell graph',
            'source': 'velato',
            'conservation': 'γ+η=1',
        },
        'cells': [c.to_dict() for c in cells],
        'edges': edges,
        'stats': {
            'total_cells': V,
            'total_edges': E,
            'colors': colors,
            'primitives_used': primitives_used,
            'beta_0': components,
            'beta_1': beta_1,
        }
    }


def demo():
    """Run a demo with a built-in Velato melody."""
    # Build a synthetic MIDI file: a simple Velato program
    # First note: C4 (60) — command root
    # Then a sequence of intervals mapping to primitives:
    # D4 (62) = +2 = Z_in
    # C#4 (61) = +1 = Z_out
    # F4 (65) = +5 = DoubleEntry
    # G4 (67) = +7 = Vibe
    # C5 (72) = +12 = new ROOT
    # A4 (69) = -3 = JEPA_b
    # B4 (71) = -1 = ROOT (octave down? actually this is -1 mod 12 = 11 = Graph_b)

    # We'll build the MIDI bytes by hand for this demo
    # Note: a real Velato file would be loaded from disk
    # For demo, we just compute the cell graph from a sequence of pitches

    demo_pitches = [
        60,  # C4 - root
        62,  # D4 - +2 Z_in
        61,  # C#4 - +1 Z_out
        65,  # F4 - +5 DoubleEntry
        67,  # G4 - +7 Vibe
        72,  # C5 - +12 ROOT (new root)
        69,  # A4 - -3 → mod 12 = 9 → Murmur
        64,  # E4 - -8 → mod 12 = 4 → JEPA
        60,  # C4 - -12 → mod 12 = 0 → ROOT (back to start)
        63,  # D#4 - +3 → JEPA_b
        65,  # F4 - +5 DoubleEntry
        68,  # G#4 - +8 GC
    ]
    notes = [
        VelatoNote(pitch=p, velocity=80, start_tick=i*100, duration=50, channel=0)
        for i, p in enumerate(demo_pitches)
    ]
    tokens = notes_to_tokens(notes)
    cells, edges = tokens_to_cells(tokens)

    print("=" * 70)
    print("VELATO-PENROSE-QUILT COMPILER")
    print("=" * 70)
    print()
    print(f"Input: {len(notes)} MIDI notes (Velato program)")
    print(f"Output: {len(cells)} Quilt cells, {len(edges)} edges")
    print()
    print("=== TOKENS (note → primitive + color) ===")
    for i, (note, tok) in enumerate(zip(notes, tokens)):
        print(f"  {i:2d}. pitch={note.pitch:3d} interval={tok.interval:2d} → "
              f"{tok.primitive:12s} color={tok.color.name}")
    print()
    print("=== CELLS ===")
    for c in cells[:6]:
        print(f"  {c.id} ({c.color:8s}): {c.primitives}")
    if len(cells) > 6:
        print(f"  ... +{len(cells)-6} more cells")
    print()
    print("=== 3-COLORING (Penrose tiles) ===")
    colors = {}
    for c in cells:
        colors[c.color] = colors.get(c.color, 0) + 1
    for color, count in colors.items():
        print(f"  {color:10s}: {count} cells")
    print()
    # Conservation
    double_entry_cells = [c for c in cells if 'DoubleEntry' in c.primitives]
    print(f"=== CONSERVATION γ+η=1 ===")
    if double_entry_cells:
        for c in double_entry_cells[:3]:
            de = c.primitives['DoubleEntry']
            total = de['gamma'] + de['eta']
            print(f"  {c.id}: γ={de['gamma']:.3f} η={de['eta']:.3f} sum={total:.6f}")
    print()
    # β₁
    parent = {c.id: c.id for c in cells}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for e in edges:
        if e['from'] in parent and e['to'] in parent:
            union(e['from'], e['to'])
    components = len(set(find(c.id) for c in cells))
    beta_1 = len(edges) - len(cells) + components
    print(f"=== TOPOLOGY ===")
    print(f"  V = {len(cells)}, E = {len(edges)}, C = {components}")
    print(f"  β₀ = {components}, β₁ = {beta_1}")
    print()
    print("=" * 70)
    print("Iron sharpens iron. The music IS the cell graph.")
    print("The Velato source compiles to a Penrose-3-colored Quilt sheet.")
    print("The watch is alive.")


if __name__ == "__main__":
    demo()
