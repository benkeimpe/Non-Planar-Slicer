"""Mesh loading, boundary loop detection, and harmonic scalar field.

The mesh is expected to be an open tube (deformed cylinder): exactly two
boundary loops, one at the bottom (planar, resting on the bed) and one at
the top (the rim, planar or wavy).

The harmonic field u is the solution of the Laplace equation on the mesh
surface with u=0 on the bottom loop and u=1 on the top rim. Its isolines
are smooth closed loops that are perpendicular to the "length" of the tube
and morph continuously from the planar base to the (possibly non-planar)
rim. These isolines are the non-planar "layers".
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class MeshError(ValueError):
    pass


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load an STL/OBJ as a single mesh with merged vertices."""
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise MeshError(f"Could not load a triangle mesh from {path!r}")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    return mesh


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Return ordered vertex-index loops for every open boundary."""
    edges = mesh.edges_sorted  # (3F, 2) sorted vertex pairs
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        raise MeshError(
            "Mesh is closed (watertight). This slicer needs an OPEN tube: "
            "remove the top/bottom caps so there are two boundary loops."
        )
    # adjacency map vertex -> neighbours along boundary
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    for v, ns in adj.items():
        if len(ns) != 2:
            raise MeshError(
                f"Non-manifold boundary at vertex {v} ({len(ns)} boundary "
                "neighbours). Clean the mesh (each boundary vertex must have "
                "exactly 2 boundary edges)."
            )
    loops = []
    visited: set[int] = set()
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, cur = None, start
        while True:
            ns = adj[cur]
            nxt = ns[0] if ns[0] != prev else ns[1]
            if nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def find_bottom_top(mesh: trimesh.Trimesh, loops: list[np.ndarray],
                    planar_tol: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Identify the bottom (bed) loop and the top rim loop by mean Z."""
    if len(loops) != 2:
        raise MeshError(
            f"Expected exactly 2 boundary loops (bottom + rim), found "
            f"{len(loops)}. The mesh must be a single open tube."
        )
    z_mean = [mesh.vertices[l][:, 2].mean() for l in loops]
    bottom, top = (loops[0], loops[1]) if z_mean[0] < z_mean[1] else (loops[1], loops[0])
    bz = mesh.vertices[bottom][:, 2]
    if bz.max() - bz.min() > planar_tol:
        raise MeshError(
            f"Bottom boundary is not planar (z range {bz.max() - bz.min():.2f} mm "
            f"> {planar_tol} mm). Orient the model with its flat open end on the bed."
        )
    return bottom, top


def harmonic_field(mesh: trimesh.Trimesh, bottom: np.ndarray,
                   top: np.ndarray) -> np.ndarray:
    """Solve Laplace u = 0 with u=0 on bottom loop, u=1 on top loop.

    Uses cotangent weights clamped to >= small positive value, which keeps
    the system an M-matrix so the discrete maximum principle holds and the
    isolines are well-ordered (no spurious extrema).
    """
    V = mesh.vertices
    F = mesh.faces
    n = len(V)

    # cotangent weights per face corner
    rows, cols, vals = [], [], []
    for c in range(3):
        i = F[:, c]
        j = F[:, (c + 1) % 3]
        k = F[:, (c + 2) % 3]
        # cot of angle at k, opposite edge (i, j)
        u = V[i] - V[k]
        v = V[j] - V[k]
        cross = np.cross(u, v)
        denom = np.linalg.norm(cross, axis=1)
        denom = np.maximum(denom, 1e-12)
        cot = (u * v).sum(axis=1) / denom
        cot = np.clip(cot, 1e-6, 1e6)  # clamp: monotone field
        w = 0.5 * cot
        rows.extend([i, j]); cols.extend([j, i]); vals.extend([w, w])
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    W = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    L = csr_matrix(
        (np.concatenate([-W.tocoo().data, d]),
         (np.concatenate([W.tocoo().row, np.arange(n)]),
          np.concatenate([W.tocoo().col, np.arange(n)]))),
        shape=(n, n))

    fixed = np.zeros(n, dtype=bool)
    fixed[bottom] = True
    fixed[top] = True
    ub = np.zeros(n)
    ub[top] = 1.0

    free = ~fixed
    A = L[free][:, free]
    b = -L[free][:, fixed] @ ub[fixed]
    u = ub.copy()
    u[free] = spsolve(A.tocsc(), b)
    # numerical safety
    u = np.clip(u, 0.0, 1.0)
    return u
