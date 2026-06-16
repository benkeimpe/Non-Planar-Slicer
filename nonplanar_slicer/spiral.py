"""Isoline extraction and continuous-spiral toolpath generation.

Pipeline:
  1. Extract many closed isoline loops of the harmonic field u.
  2. Resample every loop to J points by arc length, rotation-aligned to the
     loop below (geometric pre-alignment + windowed least-squares refinement)
     so that index j traces a smooth "column" running up the tube.
  3. Per-column cumulative arc length s_j -> normalized parameter t in [0,1].
  4. Walk a continuous spiral through the (column, t) grid: one full turn
     advances t by 1/N on every column, so the local layer height is
     column_length / N -- automatically thinner on the short side of a bend
     and thicker on the long side, meeting the rim exactly.
  5. Ramp-in (first turn rises out of the last closed base loop) and
     ramp-out (final closing turn along the rim with flow tapering to zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# isoline extraction (vectorized marching triangles)
# ---------------------------------------------------------------------------


class IsolineExtractor:
    """Extracts ordered closed isoline loops of a per-vertex scalar field."""

    def __init__(self, mesh: trimesh.Trimesh, u: np.ndarray):
        self.V = np.asarray(mesh.vertices, dtype=np.float64)
        self.F = np.asarray(mesh.faces, dtype=np.int64)
        self.u = np.asarray(u, dtype=np.float64)
        self.nV = len(self.V)
        self.fu = self.u[self.F]                       # (F, 3)
        self.fmin = self.fu.min(axis=1)
        self.fmax = self.fu.max(axis=1)

    def extract(self, level: float) -> np.ndarray:
        """Return one ordered closed loop (M, 3) at the given level.

        If the isoline has several components (shouldn't happen on a clean
        tube), the longest loop is returned.
        """
        u = self.u
        if np.any(np.abs(u - level) < 1e-12):
            level += 1e-9
        cand = (self.fmin < level) & (self.fmax > level)
        if not cand.any():
            raise ValueError(f"No isoline at level {level}")
        Fc = self.F[cand]
        s = self.fu[cand] - level                       # (n, 3)
        # edge slots: 0:(v0,v1) 1:(v1,v2) 2:(v2,v0)
        cross = s * np.roll(s, -1, axis=1) < 0          # (n, 3)
        keep = cross.sum(axis=1) == 2
        Fc, cross = Fc[keep], cross[keep]
        n = len(Fc)
        if n < 3:
            raise ValueError(f"Degenerate isoline at level {level}")

        P0 = Fc
        P1 = np.roll(Fc, -1, axis=1)
        pe = P0[cross]                                  # (2n,) face-ordered
        qe = P1[cross]
        lo = np.minimum(pe, qe).astype(np.int64)
        hi = np.maximum(pe, qe).astype(np.int64)
        enc = lo * self.nV + hi
        uniq, inv = np.unique(enc, return_inverse=True)
        ulo = (uniq // self.nV).astype(np.int64)
        uhi = (uniq % self.nV).astype(np.int64)
        t = (level - u[ulo]) / (u[uhi] - u[ulo])
        pts = self.V[ulo] + t[:, None] * (self.V[uhi] - self.V[ulo])

        fe = inv.reshape(n, 2)                          # two edge ids per face
        # edge -> (face, face) adjacency
        face_of = np.repeat(np.arange(n), 2)
        perm = np.argsort(inv, kind="stable")
        sinv = inv[perm]
        sface = face_of[perm]
        counts = np.bincount(inv, minlength=len(uniq))
        starts = np.searchsorted(sinv, np.arange(len(uniq)))
        ef0 = np.full(len(uniq), -1, dtype=np.int64)
        ef1 = np.full(len(uniq), -1, dtype=np.int64)
        ef0[:] = sface[starts]
        two = counts >= 2
        ef1[two] = sface[np.minimum(starts + 1, len(sface) - 1)][two]

        visited = np.zeros(n, dtype=bool)
        best_ids: list[int] = []
        best_len = -1.0
        while True:
            unv = np.nonzero(~visited)[0]
            if len(unv) == 0:
                break
            f0 = int(unv[0])
            ids = []
            fi = f0
            e_in = int(fe[f0, 0])
            while True:
                visited[fi] = True
                e1, e2 = int(fe[fi, 0]), int(fe[fi, 1])
                e_out = e2 if e_in == e1 else e1
                ids.append(e_out)
                a, b = ef0[e_out], ef1[e_out]
                nxt = int(b) if a == fi else int(a)
                if nxt < 0 or visited[nxt]:
                    break
                fi = nxt
                e_in = e_out
            if len(ids) >= 3:
                loop_pts = pts[ids]
                per = np.linalg.norm(
                    np.diff(np.vstack([loop_pts, loop_pts[:1]]), axis=0), axis=1).sum()
                if per > best_len:
                    best_len = per
                    best_ids = ids
        if not best_ids:
            raise ValueError(f"Isoline tracing failed at level {level}")
        return pts[best_ids]


# ---------------------------------------------------------------------------
# loop resampling + alignment
# ---------------------------------------------------------------------------


def _resample_closed(P: np.ndarray, J: int, offset: float = 0.0) -> np.ndarray:
    """Resample closed polyline P to J points uniform in arc length.

    offset is a fractional [0,1) rotation of the start point along the loop.
    """
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise ValueError("Degenerate isoline loop (zero length)")
    targets = (np.arange(J) / J + offset) % 1.0 * total
    idx = np.searchsorted(s, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    denom = np.maximum(seg[idx], 1e-12)
    frac = (targets - s[idx]) / denom
    return Q[idx] + frac[:, None] * (Q[idx + 1] - Q[idx])


def _orient_ccw_like(P: np.ndarray, ref_normal: np.ndarray) -> np.ndarray:
    """Make loop winding consistent with ref_normal (Newell's method)."""
    Q = np.vstack([P, P[:1]])
    nrm = np.cross(Q[:-1], Q[1:]).sum(axis=0)
    if np.dot(nrm, ref_normal) < 0:
        return P[::-1].copy()
    return P


def _best_alignment(prev: np.ndarray, cur: np.ndarray) -> float:
    """Fractional rotation of `cur` minimizing distance to `prev`.

    Both are (J,3) uniform-arc-length resamplings of closed loops that are
    geometrically very close (consecutive isolines ~0.1 mm apart), so the
    optimal rotation is always small: search only a window of shifts around
    zero. A global search would risk snapping to a different lobe on
    rotationally-symmetric shapes (flutes, ridges).
    """
    J = len(prev)
    W = max(4, J // 48)
    shifts = np.arange(-W, W + 1)
    idx = (np.arange(J)[None, :] + shifts[:, None]) % J     # (S, J)
    diff = cur[idx] - prev[None, :, :]                      # (S, J, 3)
    err = np.einsum("sjk,sjk->s", diff, diff)
    b = int(np.argmin(err))
    e0, e1, e2 = err[max(b - 1, 0)], err[b], err[min(b + 1, len(err) - 1)]
    denom = e0 - 2 * e1 + e2
    delta = 0.5 * (e0 - e2) / denom if abs(denom) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return float((shifts[b] + delta) % J) / J


# ---------------------------------------------------------------------------
# the surface grid
# ---------------------------------------------------------------------------


@dataclass
class SurfaceGrid:
    """K isoline loops x J columns tracing the tube surface."""

    points: np.ndarray          # (K, J, 3)
    levels: np.ndarray          # (K,) field levels
    s: np.ndarray               # (K, J) cumulative arc length per column
    col_len: np.ndarray         # (J,) total column length

    @property
    def K(self) -> int:
        return self.points.shape[0]

    @property
    def J(self) -> int:
        return self.points.shape[1]

    def sample_column(self, j: int, arc: np.ndarray) -> np.ndarray:
        """Points along column j at the given cumulative arc lengths."""
        sj = self.s[:, j]
        arc = np.clip(arc, 0.0, sj[-1])
        idx = np.searchsorted(sj, arc, side="right") - 1
        idx = np.clip(idx, 0, self.K - 2)
        seg = np.maximum(sj[idx + 1] - sj[idx], 1e-12)
        frac = ((arc - sj[idx]) / seg)[:, None]
        return self.points[idx, j] * (1 - frac) + self.points[idx + 1, j] * frac


def build_grid(extractor: IsolineExtractor, mesh: trimesh.Trimesh,
               xy_resolution: float = 0.6,
               v_resolution: float = 0.12,
               coarse_levels: int = 60,
               max_levels: int = 4000,
               progress=None) -> SurfaceGrid:
    """Extract isolines and assemble the aligned (K, J) surface grid.

    Levels are placed approximately uniform in *mean 3D spacing* (not in u),
    estimated from a coarse pass, so the grid resolves the geometry evenly.
    """
    eps = 1e-4
    # --- coarse pass: estimate mean arc height as a function of u ---------
    coarse_u = np.linspace(eps, 1 - eps, coarse_levels)
    centroids = []
    for lv in coarse_u:
        P = extractor.extract(float(lv))
        centroids.append(P.mean(axis=0))
    centroids = np.array(centroids)
    d = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])           # ~arc height vs u
    total_h = cum[-1]

    K = int(np.clip(np.ceil(total_h / v_resolution) + 1, coarse_levels, max_levels))
    targets = np.linspace(0, total_h, K)
    levels = np.interp(targets, cum, coarse_u)
    levels[0] = eps
    levels[-1] = 1 - eps

    # --- extract all loops --------------------------------------------------
    raw: list[np.ndarray] = []
    max_perim = 0.0
    up = np.array([0.0, 0.0, 1.0])
    for i, lv in enumerate(levels):
        P = extractor.extract(float(lv))
        P = _orient_ccw_like(P, up)
        raw.append(P)
        per = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()
        max_perim = max(max_perim, per)
        if progress and i % 400 == 0:
            progress(f"isolines {i}/{K}")

    J = int(np.clip(np.ceil(max_perim / xy_resolution), 48, 1440))

    # --- resample + align ---------------------------------------------------
    grid = np.empty((K, J, 3))
    grid[0] = _resample_closed(raw[0], J, 0.0)
    for i in range(1, K):
        # The traced loop starts at an arbitrary face, so first find a coarse
        # offset geometrically: the arc position of the raw vertex nearest to
        # the previous loop's column-0 point. Then refine with a small
        # windowed search.
        Q = raw[i]
        seg = np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        nearest = int(np.argmin(((Q - grid[i - 1][0]) ** 2).sum(axis=1)))
        off0 = cum[nearest] / total
        first = _resample_closed(Q, J, off0)
        off = _best_alignment(grid[i - 1], first)
        grid[i] = _resample_closed(Q, J, (off0 + off) % 1.0)
        if progress and i % 600 == 0:
            progress(f"align {i}/{K}")

    seg = np.linalg.norm(np.diff(grid, axis=0), axis=2)   # (K-1, J)
    s = np.vstack([np.zeros((1, J)), np.cumsum(seg, axis=0)])
    col_len = s[-1].copy()
    return SurfaceGrid(points=grid, levels=levels, s=s, col_len=col_len)


# ---------------------------------------------------------------------------
# spiral toolpath
# ---------------------------------------------------------------------------


@dataclass
class Toolpath:
    """A continuous extrusion path. One row per vertex of the polyline."""

    points: np.ndarray         # (M, 3)
    gap: np.ndarray            # (M,) local layer height (support distance), mm
    kind: np.ndarray           # (M,) 0=brim 1=base 2=spiral 3=close
    n_turns: int = 0
    J: int = 0
    base_count: int = 0
    brim_count: int = 0
    meta: dict = field(default_factory=dict)


def _polygon_area(p):
    x = p[:, 0]; y = p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inward_offsets(ring_xy, spacing, max_loops):
    """Concentric inward offset loops of a ring (outer -> inner), pure numpy.

    Each loop is offset inward by `spacing` along vertex normals; stops when the
    loop collapses toward the centre. No external dependencies.
    """
    cur = np.asarray(ring_xy, dtype=float)
    if len(cur) < 3 or spacing <= 0:
        return []
    out = []
    for _ in range(int(max_loops)):
        c = cur.mean(axis=0)
        nxt = np.roll(cur, -1, axis=0); prv = np.roll(cur, 1, axis=0)
        tang = nxt - prv
        nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        flip = (nrm * (cur - c)).sum(axis=1) > 0     # make normals point inward
        nrm[flip] *= -1
        cur = cur + nrm * spacing
        if len(cur) < 3 or _polygon_area(cur) < spacing * spacing:
            break
        out.append(cur.copy())
        if np.linalg.norm(cur - cur.mean(axis=0), axis=1).max() < spacing:
            break
    return out


def _ease_steepness_grid(ZT, XYf, max_deg, warnings=None):
    """Ease layer-line steepness on the target-height grid before it is resampled
    back onto the surface (so the toolpath stays exactly on the mesh).

    Baseline = a straight height ramp from base to top turn per column. The
    surface bumps are the deviation from that ramp. Per turn, where the steepest
    within-turn slope exceeds the cap, scale that turn's deviation toward the
    ramp just enough to meet the cap; the scale factors are smoothed across turns
    so the easing fades in/out gradually. Gentle turns are untouched."""
    N, J = ZT.shape
    if N < 4:
        return ZT
    cap = float(np.tan(np.radians(max_deg)))
    dj = np.linalg.norm(np.roll(XYf, -1, axis=1) - XYf, axis=2)
    t = np.linspace(0.0, 1.0, N)[:, None]
    ramp = ZT[:1, :] * (1.0 - t) + ZT[-1:, :] * t
    dev = ZT - ramp
    slope = np.abs(np.roll(ZT, -1, axis=1) - ZT) / np.maximum(dj, 1e-9)
    mx = slope.max(axis=1)
    if not (mx > cap).any():
        return ZT
    f = np.ones(N)
    over = mx > cap
    f[over] = np.clip(cap / mx[over], 0.0, 1.0)
    f[0] = 1.0; f[-1] = 1.0
    if N >= 7:
        pad = np.pad(f, 3, mode="edge")
        f = np.convolve(pad, np.ones(7) / 7.0, mode="valid")
    if warnings is not None:
        worst = float(np.degrees(np.arctan(mx.max())))
        warnings.append(f"Layer steepness eased toward ~{max_deg:.0f} deg "
                        f"(steepest layer line was ~{worst:.0f} deg).")
    return ramp + dev * f[:, None]


def _smooth_path(points, kind, J, N, strength, line_width, warnings=None):
    """Round sharp corners / steps so moves flow, by smoothing ALONG the actual
    toolpath (1-D moving average in print order). This rounds each local kink
    (e.g. the stacked-sucker zig-zag) without reinforcing it the way a grid
    average across turns would. strength 0..1 -> 0..24 passes; endpoints fixed."""
    sel = np.nonzero(kind == 2)[0]
    if len(sel) < 5:
        return
    passes = int(round(float(np.clip(strength, 0.0, 1.0)) * 24))
    if passes < 1:
        return
    P = points[sel].astype(float)
    P0 = P.copy()
    for _ in range(passes):
        P[1:-1] = P[1:-1] + 0.5 * (0.5 * (P[2:] + P[:-2]) - P[1:-1])
    points[sel] = P
    if warnings is not None:
        dev = float(np.linalg.norm(P - P0, axis=1).max())
        warnings.append(f"Path smoothing: {passes} passes "
                        f"(max deviation from surface {dev:.2f} mm).")


def generate_spiral(grid: SurfaceGrid, *,
                    layer_height: float,
                    min_layer_height: float,
                    max_layer_height: float,
                    first_layer_height: float,
                    base_loops: int = 2,
                    brim_loops: int = 0,
                    brim_spacing: float = 0.45,
                    curvature_factor: float = 1.0,
                    inward_brim_loops: int = 0,
                    inward_brim_spacing: float = 0.0,
                    max_steepness_deg: float = 90.0,
                    layer_height_factor: float = 1.0,
                    smoothing: float = 0.0,
                    line_width: float = 0.45,
                    warnings: list[str] | None = None) -> Toolpath:
    """Build the full toolpath: brim, base loops, ramped spiral, closing turn."""
    if warnings is None:
        warnings = []
    J = grid.J
    L = grid.col_len                                  # (J,)

    base_arcs = [first_layer_height + i * layer_height for i in range(base_loops)]
    start_arc = base_arcs[-1] if base_loops > 0 else 0.0
    spiral_len = L - start_arc                        # (J,)
    if np.any(spiral_len <= 0):
        raise ValueError("Part is shorter than the base loops; reduce base_loops.")

    mean_len = float(spiral_len.mean())
    # Drive N from max_layer_height so the widest column hits the user's stated
    # maximum — this gives the thickest layers possible and the strongest
    # "lensing" effect.  layer_height is still used for base-loop spacing and
    # as the flow-rate reference in gcode.py.
    N = max(2, int(np.ceil(spiral_len.max() / max_layer_height)))
    # Optionally use MORE (thinner) layers than the bare minimum so the path can
    # conform to fine geometry and the X/Y gap between layers shrinks.
    # layer_height_factor 1.0 = fewest layers / thickest (avg height near max);
    # lower = more layers (average local height approaches factor * max).
    fac = float(np.clip(layer_height_factor, 0.05, 1.0))
    if fac < 0.999:
        N = max(N, int(np.ceil(spiral_len.mean() / (max_layer_height * fac))))
    # but never thinner than min_layer_height on the longest column
    N = min(N, max(2, int(np.floor(spiral_len.max() / max(min_layer_height, 1e-6)))))
    N = max(N, 2)
    h_min_actual = spiral_len.min() / N
    h_max_actual = spiral_len.max() / N
    if h_min_actual < min_layer_height:
        warnings.append(
            f"Local layer height drops to {h_min_actual:.3f} mm "
            f"(< min {min_layer_height} mm) on the short side; "
            f"flow will be clamped by the min flow multiplier.")
    warnings.append(
        f"{N} turns; local layer height {h_min_actual:.3f}..{h_max_actual:.3f} mm "
        f"(target max {max_layer_height} mm).")

    pts_list, gap_list, kind_list = [], [], []

    def ring_at_arc(arc: float) -> np.ndarray:
        return np.vstack([grid.sample_column(j, np.array([arc]))[0] for j in range(J)])

    # --- inward brim: concentric loops spiralling in over the base -----------
    ib_len = 0
    if inward_brim_loops > 0:
        ib_arc = base_arcs[0] if base_loops > 0 else first_layer_height
        ib_ring = ring_at_arc(ib_arc)
        z0 = float(ib_ring[:, 2].mean())
        ib_spacing = inward_brim_spacing if inward_brim_spacing > 0 else line_width
        offs = _inward_offsets(ib_ring[:, :2], ib_spacing, int(inward_brim_loops))
        if offs:
            chunks = []
            for ring in offs:
                loop = np.column_stack([ring, np.full(len(ring), z0)])
                chunks.append(np.vstack([loop, loop[:1]]))
            ib = np.vstack(chunks)
            ib_len = len(ib)
            pts_list.append(ib)
            gap_list.append(np.full(ib_len, first_layer_height))
            kind_list.append(np.zeros(ib_len, dtype=np.int8))
        elif warnings is not None:
            warnings.append("Inward brim requested but the base outline was too "
                            "small to offset inward at this spacing.")

    # --- brim ----------------------------------------------------------------
    if brim_loops > 0 and base_loops > 0:
        first_ring = ring_at_arc(base_arcs[0])
        c = first_ring[:, :2].mean(axis=0)
        nxt = np.roll(first_ring[:, :2], -1, axis=0)
        prv = np.roll(first_ring[:, :2], 1, axis=0)
        tang = nxt - prv
        nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        outward = first_ring[:, :2] - c
        flip = (nrm * outward).sum(axis=1) < 0
        nrm[flip] *= -1
        z0 = first_ring[:, 2].mean()
        for b in range(brim_loops, 0, -1):       # outermost first
            xy = first_ring[:, :2] + nrm * (b * brim_spacing)
            loop = np.column_stack([xy, np.full(J, z0)])
            loop = np.vstack([loop, loop[:1]])
            pts_list.append(loop)
            gap_list.append(np.full(len(loop), first_layer_height))
            kind_list.append(np.zeros(len(loop), dtype=np.int8))

    # --- base loops ------------------------------------------------------------
    for bi, arc in enumerate(base_arcs):
        ring = ring_at_arc(arc)
        ring = np.vstack([ring, ring[:1]])
        h = first_layer_height if bi == 0 else layer_height
        pts_list.append(ring)
        gap_list.append(np.full(len(ring), h))
        kind_list.append(np.full(len(ring), 1, dtype=np.int8))

    # --- spiral ----------------------------------------------------------------
    # Curvature blending (deformation amount), done entirely ON the surface.
    #
    # For every column j the toolpath point is ALWAYS grid.sample_column(j, arc)
    # for some arc length, so it lies on the mesh surface no matter what the
    # deformation amount is.  The deformation amount only changes *which* arc
    # (i.e. how high up the column) we sample:
    #
    #   curvature_factor = 1  → arc follows the surface exactly (full non-planar,
    #                           layer lines ride over every bump; default).
    #   curvature_factor = 0  → arc follows a straight height ramp from the base
    #                           ring to the rim for that column, so local bumps
    #                           are not followed, but the layer still tilts to
    #                           match the average angle between the bottom and
    #                           top rims.
    #   in between            → smooth blend of the two.
    #
    # Because the blend is in *height* and we then resample the column back onto
    # the surface, the very top of the spiral (f→1) lands on the rim for EVERY
    # value of curvature_factor, giving a smooth handoff to the closing turn.
    total = N * J
    per_turn = spiral_len / N
    cf = float(np.clip(curvature_factor, 0.0, 1.0))
    full = cf > 1.0 - 1e-6
    steep_on = max_steepness_deg < 89.9
    karr = np.arange(N)

    # Pass 1: target height per (turn, column) + the full-deformation XY of each
    # point (used only to measure within-turn slopes for the steepness easing).
    ZT = np.empty((N, J)); XYf = np.empty((N, J, 2))
    for j in range(J):
        fj = (karr * J + j) / total
        af = start_arc + fj * spiral_len[j]
        sj = grid.s[:, j]; zj = grid.points[:, j, 2]
        z_full = np.interp(af, sj, zj)
        zb = float(np.interp(start_arc, sj, zj)); zr = float(zj[-1])
        z_ramp = (1.0 - fj) * zb + fj * zr
        ZT[:, j] = z_ramp + cf * (z_full - z_ramp)
        XYf[:, j] = grid.sample_column(j, af)[:, :2]

    # Steepness easing acts on the TARGET heights; the points are then resampled
    # back onto the surface in pass 2, so they ALWAYS lie exactly on the mesh.
    if steep_on:
        ZT = _ease_steepness_grid(ZT, XYf, float(max_steepness_deg), warnings)

    # Pass 2: for each column find the arc whose surface height matches the
    # (possibly eased) target, then sample the column there -> ON the surface.
    spiral_pts = np.empty((total, 3))
    spiral_gap = np.empty(total)
    for j in range(J):
        sj = grid.s[:, j]; zj = grid.points[:, j, 2]
        idx = karr * J + j
        if full and not steep_on:
            arc = start_arc + (idx / total) * spiral_len[j]
        else:
            z_mono = np.maximum.accumulate(zj) + np.arange(len(zj)) * 1e-9
            arc = np.interp(ZT[:, j], z_mono, sj)
            arc = np.clip(arc, start_arc, L[j])
        spiral_pts[idx] = grid.sample_column(j, arc)
        arc_prev = np.concatenate([[start_arc], arc[:-1]])
        spiral_gap[idx] = np.maximum(arc - arc_prev, 1e-6)

    pts_list.append(spiral_pts)
    gap_list.append(spiral_gap)
    kind_list.append(np.full(total, 2, dtype=np.int8))

    # --- closing turn along the rim (t = 1), flow tapering to zero -------------
    close_pts = np.empty((J + 1, 3))
    close_gap = np.empty(J + 1)
    for j in range(J):
        close_pts[j] = grid.sample_column(j, np.array([L[j]]))[0]
        close_gap[j] = per_turn[j] * (1.0 - j / J)
    close_pts[J] = close_pts[0]
    close_gap[J] = 0.0
    pts_list.append(close_pts)
    gap_list.append(close_gap)
    kind_list.append(np.full(J + 1, 3, dtype=np.int8))

    points = np.vstack(pts_list)
    gap = np.concatenate(gap_list)
    kind = np.concatenate(kind_list)

    # travel between the inward brim and the rest (avoid a drawn line from the
    # brim centre back out to the wall)
    if ib_len > 0 and ib_len < len(points):
        nxt = points[ib_len].copy()
        points = np.insert(points, ib_len, nxt, axis=0)
        gap = np.insert(gap, ib_len, gap[ib_len])
        kind = np.insert(kind, ib_len, np.int8(5))

    if smoothing > 0.0:
        _smooth_path(points, kind, J, N, float(smoothing), float(line_width), warnings)

    return Toolpath(
        points=points,
        gap=gap,
        kind=kind,
        n_turns=N, J=J, base_count=base_loops, brim_count=brim_loops,
        meta={
            "h_local_min": float(h_min_actual),
            "h_local_max": float(h_max_actual),
            "warnings": warnings,
        },
    )


def stacking_tilt(grid: SurfaceGrid) -> np.ndarray:
    """Angle (deg) between the column direction and vertical, per grid cell.

    High values mean the nozzle prints onto a steeply tilted previous layer,
    which risks collision between the nozzle cone and the part.
    """
    d = np.diff(grid.points, axis=0)              # (K-1, J, 3)
    dz = d[:, :, 2]
    dn = np.linalg.norm(d, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.degrees(np.arccos(np.clip(dz / np.maximum(dn, 1e-12), -1, 1)))
    return ang
