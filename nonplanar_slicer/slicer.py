"""Top-level slicing pipeline and CLI."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .mesh import load_mesh, boundary_loops, find_bottom_top, harmonic_field, MeshError
from .spiral import IsolineExtractor, build_grid, generate_spiral, stacking_tilt, Toolpath, SurfaceGrid
from .gcode import PrintSettings, write_gcode


def slice_mesh(mesh_path: str, settings: PrintSettings,
               verbose: bool = True) -> tuple[Toolpath, SurfaceGrid, object]:
    """Run the full pipeline. Returns (toolpath, grid, mesh)."""
    log = (lambda m: print(f"[slicer] {m}", flush=True)) if verbose else (lambda m: None)

    t0 = time.time()
    mesh = load_mesh(mesh_path)
    log(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
        f"bbox {np.round(mesh.extents, 1)} mm")

    # apply user-specified transform (rotation about centroid, then translation)
    rx = settings.transform_rotate_x
    ry = settings.transform_rotate_y
    rz = settings.transform_rotate_z
    tx = settings.transform_translate_x
    ty = settings.transform_translate_y
    tz = settings.transform_translate_z
    if any([rx, ry, rz, tx, ty, tz]):
        c = mesh.vertices.mean(axis=0)
        mesh.vertices -= c
        if rx or ry or rz:
            rxr, ryr, rzr = np.radians([rx, ry, rz])
            Rx = np.array([[1, 0, 0], [0, np.cos(rxr), -np.sin(rxr)],
                           [0, np.sin(rxr),  np.cos(rxr)]])
            Ry = np.array([[np.cos(ryr), 0, np.sin(ryr)], [0, 1, 0],
                           [-np.sin(ryr), 0, np.cos(ryr)]])
            Rz = np.array([[np.cos(rzr), -np.sin(rzr), 0],
                           [np.sin(rzr),  np.cos(rzr), 0], [0, 0, 1]])
            mesh.vertices = mesh.vertices @ (Rz @ Ry @ Rx).T
        mesh.vertices += c
        mesh.vertices[:, 0] += tx
        mesh.vertices[:, 1] += ty
        mesh.vertices[:, 2] += tz
        log(f"transform applied: rot({rx},{ry},{rz}) translate({tx},{ty},{tz})")

    # drop to bed, optionally center
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()
    if settings.center_on_bed:
        c = 0.5 * (mesh.vertices[:, :2].min(axis=0) + mesh.vertices[:, :2].max(axis=0))
        mesh.vertices[:, 0] += settings.bed_size_x / 2 - c[0]
        mesh.vertices[:, 1] += settings.bed_size_y / 2 - c[1]

    # bounds check
    mn, mx = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    margin = (settings.brim_loops + 1) * settings.resolved_line_width()
    if (mn[0] < margin or mn[1] < margin or
            mx[0] > settings.bed_size_x - margin or
            mx[1] > settings.bed_size_y - margin or
            mx[2] > settings.bed_size_z):
        raise MeshError(
            f"Model (+brim) exceeds build volume "
            f"{settings.bed_size_x}x{settings.bed_size_y}x{settings.bed_size_z}: "
            f"bounds {np.round(mn, 1)}..{np.round(mx, 1)}")

    # ignore any flat bottom surface (faces coplanar with the bed); we slice the
    # outer walls only, so closed-bottom models are handled by dropping the base.
    zmin = mesh.vertices[:, 2].min()
    fz = mesh.vertices[mesh.faces][:, :, 2]
    nz = mesh.face_normals[:, 2]
    bottom_face = (fz <= zmin + settings.bottom_strip_tol).all(axis=1) & (np.abs(nz) > 0.7)
    if bottom_face.any() and not bottom_face.all():
        mesh.update_faces(~bottom_face)
        mesh.remove_unreferenced_vertices()
        log(f"ignored {int(bottom_face.sum())} flat bottom face(s) coplanar with the bed")

    loops = boundary_loops(mesh)
    bottom, top = find_bottom_top(mesh, loops)
    log(f"boundaries: bottom {len(bottom)} verts (z~{mesh.vertices[bottom][:,2].mean():.2f}), "
        f"top {len(top)} verts (z {mesh.vertices[top][:,2].min():.1f}..{mesh.vertices[top][:,2].max():.1f})")

    u = harmonic_field(mesh, bottom, top)
    log(f"harmonic field solved ({time.time()-t0:.1f}s)")

    extractor = IsolineExtractor(mesh, u)
    grid = build_grid(extractor, mesh,
                      xy_resolution=settings.xy_resolution,
                      v_resolution=settings.v_resolution,
                      progress=log if verbose else None)
    log(f"grid: {grid.K} loops x {grid.J} columns ({time.time()-t0:.1f}s)")

    warnings: list[str] = []
    tilt = stacking_tilt(grid)
    tmax = float(np.nanmax(tilt))
    if tmax > settings.max_tilt_warn:
        frac = float((tilt > settings.max_tilt_warn).mean()) * 100
        warnings.append(
            f"Layer stacking tilts up to {tmax:.0f} deg from vertical "
            f"({frac:.1f}% of surface beyond {settings.max_tilt_warn} deg) - "
            f"check nozzle clearance on steep regions.")

    tp = generate_spiral(
        grid,
        layer_height=settings.layer_height,
        min_layer_height=settings.min_layer_height,
        max_layer_height=settings.max_layer_height,
        first_layer_height=settings.first_layer_height,
        base_loops=settings.base_loops,
        brim_loops=settings.brim_loops,
        brim_spacing=settings.resolved_line_width() * 0.9,
        curvature_factor=settings.curvature_factor,
        inward_brim_loops=settings.inward_brim_loops,
        inward_brim_spacing=settings.inward_brim_spacing,
        max_steepness_deg=settings.max_steepness_deg,
        layer_height_factor=settings.layer_height_factor,
        smoothing=settings.smoothing,
        line_width=settings.resolved_line_width(),
        warnings=warnings,
    )
    log(f"toolpath: {len(tp.points)} points, {tp.n_turns} turns ({time.time()-t0:.1f}s)")
    for w in warnings:
        log(f"WARNING: {w}")
    return tp, grid, mesh


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="nonplanar_slicer",
        description="Non-planar continuous-spiral (vase mode) slicer for open tube meshes.")
    ap.add_argument("mesh", help="STL or OBJ of an open tube (planar bottom rim on the bed)")
    ap.add_argument("-o", "--output", default=None, help="output .gcode path")
    ap.add_argument("-c", "--config", default=None, help="JSON settings file")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="after slicing, check surface match + self-overlap")
    # quick overrides for the most common settings
    s_default = PrintSettings()
    for name in ("nozzle_diameter", "line_width", "layer_height", "min_layer_height",
                 "max_layer_height", "first_layer_height", "print_speed",
                 "first_layer_speed", "nozzle_temp", "bed_temp", "fan_percent",
                 "flow_rate", "min_flow_multiplier", "max_flow_multiplier",
                 "base_loops", "brim_loops", "xy_resolution", "max_volumetric_speed"):
        ap.add_argument(f"--{name.replace('_','-')}", type=float, default=None,
                        help=f"default {getattr(s_default, name)}")
    args = ap.parse_args(argv)

    settings = PrintSettings.from_json(args.config) if args.config else PrintSettings()
    for name in vars(args):
        if name in ("mesh", "output", "config", "quiet", "verify"):
            continue
        v = getattr(args, name)
        if v is not None:
            cur = getattr(settings, name)
            setattr(settings, name, type(cur)(v))

    out = args.output or (args.mesh.rsplit(".", 1)[0] + ".gcode")
    try:
        tp, grid, mesh = slice_mesh(args.mesh, settings, verbose=not args.quiet)
    except MeshError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stats = write_gcode(out, tp, settings, model_name=args.mesh)
    print(f"wrote {out}")
    print(f"  {stats['segments']} segments, {stats['filament_m']} m "
          f"({stats['filament_g']} g), est. {stats['print_time_h']} h, "
          f"z max {stats['z_max']:.1f} mm")
    if args.verify:
        from .verify import verify
        ok, report = verify(tp, mesh, gcode_path=out, settings=settings)
        print("--- verification ---")
        for line in report:
            print("  " + line)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
