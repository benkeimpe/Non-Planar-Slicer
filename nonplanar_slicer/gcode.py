"""Print settings and Klipper-flavoured GCODE generation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .spiral import Toolpath


@dataclass
class PrintSettings:
    # geometry / extrusion
    nozzle_diameter: float = 0.6        # mm
    line_width: float = 0.0             # mm; 0 -> 1.15 * nozzle
    layer_height: float = 0.25          # nominal, mm
    min_layer_height: float = 0.08      # mm
    max_layer_height: float = 0.45      # mm
    first_layer_height: float = 0.3     # mm
    filament_diameter: float = 1.75     # mm

    # flow
    flow_rate: float = 1.0              # global extrusion multiplier
    min_flow_multiplier: float = 0.25   # clamp of (local_h / nominal_h)
    max_flow_multiplier: float = 2.0
    max_volumetric_speed: float = 9.0   # mm^3/s (PETG-ish)

    # speeds (mm/s)
    print_speed: float = 35.0
    first_layer_speed: float = 18.0
    travel_speed: float = 150.0

    # temperatures (PETG defaults)
    nozzle_temp: int = 240
    bed_temp: int = 80
    fan_percent: float = 30.0           # 0..100

    # path structure
    base_loops: int = 2
    brim_loops: int = 6
    xy_resolution: float = 0.6          # mm, segment length around loops
    v_resolution: float = 0.12          # mm, isoline grid spacing

    # machine
    bed_size_x: float = 500.0
    bed_size_y: float = 500.0
    bed_size_z: float = 500.0
    center_on_bed: bool = True
    max_tilt_warn: float = 40.0         # deg, nozzle-collision warning

    # transform (applied to mesh before bed-drop/centering)
    transform_rotate_x: float = 0.0   # degrees, Euler XYZ about model centroid
    transform_rotate_y: float = 0.0
    transform_rotate_z: float = 0.0
    transform_translate_x: float = 0.0  # mm offset after centering
    transform_translate_y: float = 0.0
    transform_translate_z: float = 0.0

    # non-planar
    curvature_factor: float = 1.0  # 0 = flat layers, 1 = full surface-following

    # bottom / brim
    bottom_strip_tol: float = 0.5        # mm; horizontal faces within this of Z=0 are ignored
    inward_brim_loops: int = 0           # concentric loops spiralling inward over the base (0 = off)
    inward_brim_spacing: float = 0.0     # mm between inward-brim lines (0 = line width)

    # printability
    max_steepness_deg: float = 90.0      # cap on layer-line slope from horizontal (90 = off)
    layer_height_factor: float = 1.0     # 1 = fewest/thickest layers; lower = more, thinner layers
    smoothing: float = 0.0               # 0 = off; rounds sharp corners (stays within ~1 line width)

    # gcode
    retract_mm: float = 1.2
    retract_speed: float = 35.0         # mm/s
    z_hop_end: float = 10.0
    start_gcode: str = (
        "G28\n"
        "G90\n"
        "M83\n"
    )
    end_gcode: str = (
        "M104 S0\n"
        "M140 S0\n"
        "M107\n"
        "M84\n"
    )

    def resolved_line_width(self) -> float:
        return self.line_width if self.line_width > 0 else round(self.nozzle_diameter * 1.15, 3)

    @classmethod
    def from_json(cls, path: str) -> "PrintSettings":
        with open(path) as f:
            data = json.load(f)
        s = cls()
        unknown = [k for k in data if not hasattr(s, k)]
        if unknown:
            raise ValueError(f"Unknown settings: {unknown}")
        for k, v in data.items():
            setattr(s, k, type(getattr(s, k))(v))
        return s

    def to_dict(self) -> dict:
        return asdict(self)


def write_gcode(path: str, tp: Toolpath, settings: PrintSettings,
                model_name: str = "") -> dict:
    """Write the toolpath as Klipper GCODE. Returns stats dict."""
    s = settings
    width = s.resolved_line_width()
    fil_area = math.pi * (s.filament_diameter / 2.0) ** 2

    P = tp.points
    gap = tp.gap
    kind = tp.kind
    n = len(P)

    # per-segment quantities (segment i: P[i-1] -> P[i])
    d = np.diff(P, axis=0)
    dist = np.linalg.norm(d, axis=1)
    seg_gap = 0.5 * (gap[1:] + gap[:-1])
    seg_kind = np.maximum(kind[1:], kind[:-1])

    # flow multiplier clamped
    mult = seg_gap / s.layer_height
    mult = np.clip(mult, s.min_flow_multiplier, s.max_flow_multiplier)
    eff_h = mult * s.layer_height
    # taper-out on the closing turn: let flow go BELOW the min clamp to zero
    closing = seg_kind == 3
    raw_mult = seg_gap / s.layer_height
    eff_h[closing] = np.minimum(eff_h[closing], np.maximum(raw_mult[closing], 0.0) * s.layer_height)

    vol = dist * width * eff_h * s.flow_rate          # mm^3
    e = vol / fil_area                                # mm of filament

    # speeds: first layer for brim/base-loop-0; volumetric clamp elsewhere
    speed = np.full(n - 1, s.print_speed)
    first_layerish = (seg_kind == 0) | ((seg_kind == 1) & (seg_gap >= s.first_layer_height - 1e-9) & (seg_gap <= s.first_layer_height + 1e-9))
    speed[first_layerish] = s.first_layer_speed
    with np.errstate(divide="ignore", invalid="ignore"):
        vmax = s.max_volumetric_speed / np.maximum(width * eff_h, 1e-9)
    speed = np.minimum(speed, vmax)
    speed = np.maximum(speed, 1.0)

    # stop extruding once the closing-turn flow has tapered below threshold
    cutoff = s.min_flow_multiplier * 0.4
    dead = closing & (raw_mult < cutoff)
    dead = dead | (seg_kind == 5)        # kind 5 = travel move (no extrusion)

    lines: list[str] = []
    a = lines.append
    a("; generated by nonplanar_slicer (continuous-spiral non-planar vase mode)")
    a(f"; model: {model_name}")
    a(f"; turns: {tp.n_turns}  points/turn: {tp.J}")
    a(f"; local layer height: {tp.meta.get('h_local_min', 0):.3f}..{tp.meta.get('h_local_max', 0):.3f} mm")
    a("; SETTINGS_JSON: " + json.dumps(s.to_dict()))
    for w in tp.meta.get("warnings", []):
        a(f"; warning: {w}")
    a("")
    a(f"M140 S{s.bed_temp}")
    a(f"M104 S{s.nozzle_temp}")
    a(f"M190 S{s.bed_temp}")
    a(f"M109 S{s.nozzle_temp}")
    a(s.start_gcode.strip())
    a(f"M106 S{int(round(s.fan_percent * 2.55))}")
    a("G92 E0")

    # travel to start
    x0, y0, z0 = P[0]
    a(f"G0 F{int(s.travel_speed * 60)} X{x0:.3f} Y{y0:.3f} Z{z0 + 1.0:.3f}")
    a(f"G0 Z{z0:.3f}")
    a(f"G1 E{s.retract_mm:.3f} F{int(s.retract_speed * 60)}  ; prime")

    total_e = 0.0
    total_t = (
        0.0
    )
    last_f = -1.0
    fan_done = False
    for i in range(n - 1):
        if dist[i] < 1e-9:
            continue
        x, y, z = P[i + 1]
        f_mmmin = speed[i] * 60.0
        if dead[i]:
            # travel (no extrusion) along the remainder of the closing turn
            a(f"G0 X{x:.3f} Y{y:.3f} Z{z:.3f}")
            continue
        cmd = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} E{e[i]:.5f}"
        if abs(f_mmmin - last_f) > 0.5:
            cmd += f" F{f_mmmin:.0f}"
            last_f = f_mmmin
        a(cmd)
        total_e += e[i]
        total_t += dist[i] / speed[i]

    a(f"G1 E-{s.retract_mm:.3f} F{int(s.retract_speed * 60)}")
    a("G91")
    a(f"G0 Z{s.z_hop_end:.2f} F600")
    a("G90")
    a(s.end_gcode.strip())

    fil_len_m = total_e / 1000.0
    fil_g = total_e * fil_area * 1.27 / 1000.0   # PETG ~1.27 g/cm^3
    stats = {
        "segments": int(n - 1),
        "filament_mm": round(total_e, 1),
        "filament_m": round(fil_len_m, 2),
        "filament_g": round(fil_g, 1),
        "print_time_s": round(total_t, 0),
        "print_time_h": round(total_t / 3600.0, 2),
        "z_max": float(P[:, 2].max()),
    }
    lines.insert(5, f"; filament: {stats['filament_m']} m ({stats['filament_g']} g)   est. time: {stats['print_time_h']} h")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return stats
