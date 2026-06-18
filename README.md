# Non-Planar Spiral Slicer

A non-planar **continuous-spiral (vase-mode) slicer** for open tube meshes —
single-wall shapes that are open at the top and bottom, such as vases and
deformed cylinders.

The layer lines stay **perpendicular to the length of the part** the whole way
up: they start planar on the bed and smoothly tilt/morph until they meet the
top rim exactly, even if the rim is wavy or cut at an angle. Layer height
varies continuously around and along the part to make that possible, and flow
is scaled to match.

## How it works

1. **Harmonic field** — the mesh must be an open tube (two boundary loops).
   The slicer solves the Laplace equation over the surface with `u = 0` on the
   bottom loop and `u = 1` on the rim. Isolines of `u` are smooth closed loops
   perpendicular to the tube's length, morphing from the planar base to the
   rim shape. These are the non-planar "layers".
2. **Surface grid** — a few thousand isolines are extracted, resampled, and
   rotation-aligned so that vertical "columns" trace the surface. Each
   column's arc length is the local "height" of the part.
3. **Continuous spiral** — one full turn advances every column by
   `column_length / N_turns`, so the path spirals from base to rim with no
   z-seam. Where the part curves, the inside of a bend automatically gets
   thinner layers and the outside thicker ones (bounded by your min/max layer
   height).
4. **Ramp in/out** — the spiral rises continuously out of the last closed base
   loop (flow grows from zero over the first turn), and a final closing turn
   runs along the rim with flow tapering to zero, so there is no step at the
   start or end of the spiral.
5. **Flow** — extrusion per segment is proportional to the *local* layer
   height, clamped by your min/max flow multipliers, with a volumetric speed
   cap (the printer slows down instead of over-extruding on thick layers).

## Get it (Windows)

1. **Download:** on the GitHub page click **Code ▸ Download ZIP**, then
   right-click the ZIP ▸ **Extract All**. (Or `git clone` the repo.)
2. **Python:** install Python 3.10+ from [python.org](https://www.python.org/downloads/),
   making sure **"Add Python to PATH"** is ticked in the installer.

That's everything you need; the steps below do the rest.

## Requirements

Installed for you by `Setup - install requirements.bat`, or manually with:

```
pip install -r requirements.txt
```

(numpy, scipy, trimesh, rtree — Python 3.10+.)

## Usage — GUI (recommended)

1. First time only: double-click **`Setup - install requirements.bat`**.
2. Double-click **`RUN SLICER.bat`**. The app opens in your browser.
3. Drag your STL/OBJ into the drop zone, adjust settings, click **Slice**.
   The toolpath appears in the built-in 3D viewer (colored by local layer
   height); gcode is saved to the `output/` folder (created automatically next
   to the app) and can also be downloaded with the button. Settings persist
   between sessions (`gui_settings.json`).

An example model (`Example_Vase.obj`) is included to try it out, and an example
print head (`Example_Nozzle.stl`) is loaded as the nozzle automatically — swap
in your own with the nozzle drop zone.

### 3D viewer controls

- **Orbit / zoom**: drag empty space / scroll. A **Zoom sensitivity** slider in
  View options helps on a trackpad. **Color by** layer height / Z / speed / flow.
- **Print progress** slider: scrub the printed portion up and down the model.
- **Layer Conformity**: balances even, planar layers (calm, but X/Y gaps open on
  overhangs) against hugging the surface (each layer sits on the one below, so
  beads stay supported). Always stays on the surface and meets the rim.
- **Layer Density**: how hard to maximise layer height. 100% = fewest, thickest
  layers; lower it to add more, thinner layers so the path conforms better and
  the gaps between layers shrink (kept within your min/max layer height).
- **Layer Steepness Limit**: optional cap that eases layer lines steeper than
  your angle back toward even (best where steepness comes from local bumps).
- **Path Smoothing**: rounds sharp corners / steps for flowy moves (helps
  stacked-feature zig-zags); higher cuts corners a little more, so it drifts
  slightly off the surface.
- **Nozzle model**: drop in an STL/OBJ of your print head, then **click-drag the
  nozzle in the 3D view and circle it** — it walks along the toolpath so you can
  spiral it up or down to check clearance. A default nozzle loads automatically.

## Usage — command line

```bash
# slice with defaults (0.6 mm nozzle, Klipper-flavoured gcode)
python -m nonplanar_slicer model.obj -o model.gcode

# with a settings file and/or quick overrides
python -m nonplanar_slicer model.stl -c my_settings.json \
    --layer-height 0.3 --nozzle-temp 245 --brim-loops 8

# verify the result (surface match + self-overlap checks)
python -m nonplanar_slicer model.obj -o model.gcode --verify
```

Settings: the GUI writes your choices to `gui_settings.json`, which you can
also pass on the command line with `-c`. Anything not in your JSON keeps its
default; CLI flags override the JSON. Key settings:

| setting | meaning |
|---|---|
| `layer_height` / `min_` / `max_layer_height` | nominal layer height and hard bounds for the adaptive spacing |
| `first_layer_height`, `base_loops`, `brim_loops` | bed adhesion: closed planar-ish loops + brim before the spiral starts |
| `min_/max_flow_multiplier` | clamp on flow as local layer height deviates from nominal |
| `max_volumetric_speed` | mm³/s cap; speed drops on thick layers |
| `xy_resolution` / `v_resolution` | toolpath segment length / isoline grid spacing |
| `max_tilt_warn` | warn when layers tilt more than this from vertical (nozzle clearance) |
| `curvature_factor` | layer conformity 0..1 (0 = planar/even, 1 = hug the surface) |
| `layer_height_factor` | 1 = fewest/thickest layers; lower = more, thinner layers |
| `max_steepness_deg` | cap on layer-line slope from horizontal (90 = off) |
| `smoothing` | path smoothing 0..1 (rounds sharp corners; 0 = off) |
| `inward_brim_loops` / `inward_brim_spacing` | concentric loops filling inward over the base (0 = off) |

## Input mesh rules

- **Single-wall, open top.** The walls are what get sliced. A **closed/flat
  bottom is fine** — any face coplanar with the bed (Z=0) is automatically
  ignored and the slice starts from the outer edge of the base.
- **Bottom planar on the bed**; the model is dropped to z=0 and centered on the
  bed automatically.
- The top rim can be planar, wavy, or an angled/perpendicular cut.
- STL or OBJ. Vertices are merged automatically.

## Printing notes

- The defaults assume a 0.6 mm nozzle with a 0.7 mm line width; wide-and-slow
  generally gives the cleanest walls. Set temperature, speed, fan, and flow for
  your own material and machine in the settings panel.
- Watch the tilt warning: beyond ~40–45° from vertical the side of the nozzle
  can graze the part on steeply curved regions. A long or conical nozzle helps.
- Load your print head as the nozzle model and drag it along the toolpath to
  check clearance before committing to a print.

## Project layout

```
RUN SLICER.bat                  double-click to start
Setup - install requirements.bat   first-time dependency install
app.py                          local web server + GUI backend
gui.html                        the GUI / 3D viewer (served by app.py)
gui_settings.json               your saved settings
requirements.txt                Python dependencies
Example_Vase.obj                example model to slice
Example_Nozzle.stl              example print head (auto-loaded as the nozzle)
nonplanar_slicer/
  mesh.py     mesh loading, boundary loops, harmonic field
  spiral.py   isoline extraction, surface grid, spiral generation
  gcode.py    PrintSettings + Klipper gcode writer
  slicer.py   pipeline + CLI
  verify.py   surface-match & self-overlap checks
```

`output/` is created automatically next to the app the first time you slice.

## A note on filament

Please consider printing with recycled filament instead of new plastic — the
slicer asks once when it starts, and it would love a "yes". 🌍

## License

Free to use, modify, and share. If you build on it, a link back is appreciated.
If you want to offer me support, I'd definitely appreciate - it's not easy being an artist.
My instagram is a good way to reach me or purchase some of my things :)
https://www.instagram.com/_manifestdesign/
