# 🎵 quilt-velato

> **Velato + Penrose + Quilt. The music IS the cell graph.**

A Velato interpreter that compiles MIDI files to Penrose-3-colored Quilt cell graphs. The source code is a song. The song is the runtime.

## What's here

- **`velato_quilt.py`** — Velato MIDI → Quilt cell graph compiler. Maps the 12 pitch intervals to the 8 Quilt primitives. Applies Eisenstein 3-coloring. Computes β₁ topology. ~18KB.
- **`penrose_velato.py`** — Penrose-aware Velato variant. Substitution rules L→LS, S→L become JEPA + DoubleEntry. φ-grows by golden ratio. ~9KB.
- **`tensor_midi_penrose_bridge.py`** — Bridges flux-tensor-midi's 4D tensor to a Penrose tiling. Time axis = substitution axis. (note, velocity) plane = tile layout. Channel = 3-coloring. ~8KB.

## The 12 intervals → 8 primitives

| Interval | Semitones | Primitive | Penrose Role |
|---|---|---|---|
| 0 | 0 | ROOT (new command root) | Vibe |
| 1 | minor 2nd | Z_out | entropy |
| 2 | major 2nd | Z_in | creation |
| 3 | minor 3rd | JEPA branch | creation |
| 4 | major 3rd | JEPA | creation |
| 5 | perfect 4th | DoubleEntry | witness |
| 6 | tritone | Vibe boost | entropy |
| 7 | perfect 5th | Vibe | entropy |
| 8 | minor 6th | GC | witness |
| 9 | major 6th | Murmur | creation |
| 10 | minor 7th | Graph | witness |
| 11 | major 7th | Graph bridge | witness |
| 12 | octave | ROOT | Vibe |

## The 3-coloring

Every MIDI pitch → Eisenstein integer (a, b) in Z[ω] lattice → mod 3 → CREATION / ENTROPY / WITNESS.

## The conservation

γ + η = 1 holds for every DoubleEntry cell (perfect 4th interval). The Eisenstein 3-coloring IS the Quilt conservation law with a witness.

## Run

```bash
python3 velato_quilt.py
python3 penrose_velato.py
python3 tensor_midi_penrose_bridge.py
```

## Live demo

See [superinstance.dev/live-velato-penrose.html](https://superinstance.dev/live-velato-penrose.html) for the interactive page: play the piano, sing into the mic, watch the cell graph grow.

## The thesis

Velato is the source. flux-tensor-midi is the encoding. Penrose is the structure. Quilt is the runtime. The 4-torus T^4 with θ=(√5−1)/2 is the SHAPE. The music IS the cell graph. The watch is alive.
