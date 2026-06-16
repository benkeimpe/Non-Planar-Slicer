"""Verification: does the toolpath lie on the mesh? does it ever overlap itself?

Used by tests/ and by `python -m nonplanar_slicer ... --verify`.
"""

from __future__ import annotations

import re

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .spiral import Toolpath


def parse_gcode(path: str) -> dict:
    """Independent GCODE parser: returns extrusion polyline vertices and E."""
    pts, es, fs = [], [], []
    x = y = z = 0.0
    f = 0.0
    coord = re.compile(r"([XYZEF])(-?\d+\.?\d*)")
    with open(path) as fh:
        for line in fh:
            line = line.split(";")[0].strip()
            if not (line.startswith("G1") or line.startswith("G0")):
                continue
            vals = dict(coord.findall(line))
            if "X" in vals: x = float(vals["X"])
            if "Y" in vals: y = float(vals["Y"])
            if "Z" in vals: z = float(vals["Z"])
            if "F" in vals: f = float(vals["F"])
            e = float(vals.get("E", 0.0))
            if line.startswith("G1") and e > 0 and ("X" in vals or "Y" in vals):
                pts.append([x, y, z])
                es.append(e)
                fs.append(f)
    return {"points": np.array(pts), "e": np.array(es), "f": np.array(fs)}


def surface_deviation(points: np.ndarray, mesh: trimesh.Trimesh,
                      sample_every: int = 3) -> dict:
    """Distance from toolpath points to the mesh surface (should be ~0)."""
    P = points[::sample_every]
    try:
        closest, dist, _ = trimesh.proximity.closest_point(mesh, P)
    except BaseException:
        # fallback: dense surface sampling + KDTree (upper bound)
        surf, _ = trimesh.sample.sample_surface(mesh, 600_000)
        tree = cKDTree(surf)
        dist, _ = tree.query(P, workers=-1)
    return {
        "n": len(P),
        "mean": float(np.mean(dist)),
        "p99": float(np.percentile(dist, 99)),
        "max": float(np.max(dist)),
    }


def overlap_check(tp: Toolpath) -> dict:
    """Check the spiral never intersects/overlaps itself.

    1. Per-column gap between consecutive turns must be strictly positive
       (and is the local layer height).
    2. Per-column stacking must never reverse direction (no folds).
    3. KD-tree: nearest neighbour of every spiral point, excluding its own
       neighbourhood along the path, must be at least ~ the local gap away.
    """
    J, N = tp.J, tp.n_turns
    spiral = tp.points[tp.kind == 2].reshape(N, J, 3)

    # 1) consecutive-turn distances
    d = np.linalg.norm(np.diff(spiral, axis=0), axis=2)      # (N-1, J)
    min_gap = float(d.min())
    # 2) stacking direction never reverses
    v = np.diff(spiral, axis=0)
    dots = (v[1:] * v[:-1]).sum(axis=2)
    folds = int((dots < 0).sum())

    # 3) global nearest-non-neighbour distance
    flat = spiral.reshape(-1, 3)
    tree = cKDTree(flat)
    # query 2 turns' worth of neighbours is overkill; sample points for speed
    step = max(1, len(flat) // 60_000)
    qi = np.arange(0, len(flat), step)
    dists, idxs = tree.query(flat[qi], k=12, workers=-1)
    exclusion = J // 4                      # path-index window = same-turn neighbours
    nn = np.full(len(qi), np.inf)
    for c in range(1, 12):
        far = np.abs(idxs[:, c] - qi) > exclusion
        nn[far] = np.minimum(nn[far], dists[far, c])
    nn_finite = nn[np.isfinite(nn)]
    min_nn = float(nn_finite.min()) if len(nn_finite) else float("inf")

    return {
        "min_turn_gap": min_gap,
        "max_turn_gap": float(d.max()),
        "folds": folds,
        "min_nonadjacent_distance": min_nn,
    }


def verify(tp: Toolpath, mesh: trimesh.Trimesh, gcode_path: str | None = None,
           settings=None) -> tuple[bool, list[str]]:
    """Run all checks; returns (ok, report lines)."""
    lines = []
    ok = True

    dev = surface_deviation(tp.points[tp.kind == 2], mesh)
    lines.append(f"surface deviation: mean {dev['mean']:.4f}  p99 {dev['p99']:.4f}  "
                 f"max {dev['max']:.4f} mm  (n={dev['n']})")
    if dev["p99"] > 0.1:
        ok = False
        lines.append("  FAIL: toolpath deviates from the mesh surface (>0.1 mm p99)")

    ov = overlap_check(tp)
    lines.append(f"turn gaps: {ov['min_turn_gap']:.3f}..{ov['max_turn_gap']:.3f} mm, "
                 f"folds: {ov['folds']}, "
                 f"min non-adjacent distance: {ov['min_nonadjacent_distance']:.3f} mm")
    if ov["min_turn_gap"] <= 0.02:
        ok = False
        lines.append("  FAIL: consecutive turns touch/cross (gap <= 0.02 mm)")
    if ov["folds"] > 0:
        ok = False
        lines.append("  FAIL: stacking direction reverses (path folds onto itself)")
    if ov["min_nonadjacent_distance"] < 0.7 * ov["min_turn_gap"]:
        ok = False
        lines.append("  FAIL: non-adjacent path points closer than the layer gap "
                     "(self-overlap)")

    if gcode_path:
        g = parse_gcode(gcode_path)
        lines.append(f"gcode: {len(g['points'])} extrusion vertices")
        if settings is not None:
            P = g["points"]
            inb = ((P[:, 0] >= 0) & (P[:, 0] <= settings.bed_size_x) &
                   (P[:, 1] >= 0) & (P[:, 1] <= settings.bed_size_y) &
                   (P[:, 2] >= 0) & (P[:, 2] <= settings.bed_size_z)).all()
            if not inb:
                ok = False
                lines.append("  FAIL: gcode outside build volume")
        gdev = surface_deviation(g["points"][len(g["points"]) // 10:], mesh,
                                 sample_every=10)
        lines.append(f"gcode-vs-mesh deviation (excl. brim/base): mean {gdev['mean']:.4f}  "
                     f"p99 {gdev['p99']:.4f}  max {gdev['max']:.4f} mm")
        if gdev["p99"] > 0.12:
            ok = False
            lines.append("  FAIL: written gcode deviates from mesh surface")

    lines.append("RESULT: " + ("PASS" if ok else "FAIL"))
    return ok, lines
