"""Non-Planar Spiral Slicer - local GUI app.

Double-click "RUN SLICER.bat" (or run: python app.py). It starts a local
web server and opens the GUI in your browser: drag in an STL/OBJ, adjust
settings, click Slice, inspect the toolpath in 3D, download the gcode.

Everything runs locally; nothing leaves your machine.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if getattr(sys, "frozen", False):           # running as a PyInstaller exe
    ROOT = sys._MEIPASS                       # bundled resources (gui.html)
    APPDIR = os.path.dirname(sys.executable)  # settings/output next to the exe
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    APPDIR = ROOT
sys.path.insert(0, ROOT)

from nonplanar_slicer.gcode import PrintSettings, write_gcode      # noqa: E402
from nonplanar_slicer.slicer import slice_mesh                     # noqa: E402
from nonplanar_slicer.mesh import MeshError                        # noqa: E402

GUI_SETTINGS = os.path.join(APPDIR, "gui_settings.json")
OUTPUT_DIR = os.path.join(APPDIR, "output")
# runtime markers (git-ignored): one suppresses the recycled-filament prompt,
# the other disables the slicer entirely.
ACK_FLAG = os.path.join(APPDIR, ".recycled_ack")
LOCK_FLAG = os.path.join(APPDIR, ".disabled")

DISABLED_HTML = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<title>Non-Planar Spiral Slicer</title><style>"
    "html,body{height:100%;margin:0;background:#101216;color:#dde3ea;"
    "font:16px/1.6 system-ui,sans-serif;display:flex;align-items:center;"
    "justify-content:center;text-align:center}div{max-width:460px;padding:24px}"
    "h1{color:#ff6b6b;font-size:24px;margin:0 0 12px}</style></head><body><div>"
    "<h1>this tool is not for you</h1>"
    "<p>You said you genuinely don't care about using recycled filament, "
    "so this slicer has turned itself off.</p></div></body></html>"
).encode()

_busy = threading.Lock()
_log: list[str] = []


class _LogWriter(io.TextIOBase):
    def write(self, s):
        s = s.strip()
        if s:
            _log.append(s)
        return len(s)


def _settings_dict() -> dict:
    if os.path.exists(GUI_SETTINGS):
        try:
            saved = json.load(open(GUI_SETTINGS))
            base = PrintSettings().to_dict()
            base.update({k: v for k, v in saved.items() if k in base})
            return base
        except Exception:
            pass
    return PrintSettings().to_dict()


def _make_settings(data: dict) -> PrintSettings:
    s = PrintSettings()
    for f in fields(PrintSettings):
        if f.name in data and data[f.name] is not None:
            cur = getattr(s, f.name)
            try:
                setattr(s, f.name, type(cur)(data[f.name]))
            except (TypeError, ValueError):
                pass
    return s


def do_slice(mesh_bytes: bytes, filename: str, settings_data: dict,
             run_verify: bool) -> dict:
    _log.clear()
    settings = _make_settings(settings_data)
    suffix = ".stl" if filename.lower().endswith(".stl") else ".obj"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(mesh_bytes)
        tmp = tf.name
    try:
        with contextlib.redirect_stdout(_LogWriter()):
            tp, grid, mesh = slice_mesh(tmp, settings, verbose=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            base = os.path.splitext(os.path.basename(filename))[0]
            out_path = os.path.join(OUTPUT_DIR, base + ".gcode")
            stats = write_gcode(out_path, tp, settings, model_name=filename)
            result = {
                "ok": True,
                "stats": stats,
                "warnings": tp.meta.get("warnings", []),
                "output_path": out_path,
                "gcode": open(out_path).read(),
            }
            if run_verify:
                _log.append("verifying (surface match + self-overlap)...")
                from nonplanar_slicer.verify import verify
                vok, report = verify(tp, mesh, gcode_path=out_path,
                                     settings=settings)
                result["verify_ok"] = vok
                result["verify_report"] = report
        return result
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                       # silence request spam
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: bytes, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            if os.path.exists(LOCK_FLAG):       # disabled: serve the gag page
                self._html(DISABLED_HTML)
                return
            try:
                body = open(os.path.join(ROOT, "gui.html"), "rb").read()
            except OSError:
                self.send_error(500, "gui.html not found")
                return
            self._html(body)
        elif path == "/recycled_status":
            self._json({"ack": os.path.exists(ACK_FLAG),
                        "disabled": os.path.exists(LOCK_FLAG)})
        elif path == "/default_nozzle":
            p = os.path.join(ROOT, "Example_Nozzle.stl")
            if not os.path.exists(p):
                self.send_error(404)
                return
            body = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/settings":
            self._json(_settings_dict())
        elif path == "/status":
            self._json({"log": _log[-200:], "busy": _busy.locked()})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if path == "/recycled_ack":
            try:
                open(ACK_FLAG, "w").close()      # don't ask again
            except OSError:
                pass
            self._json({"ok": True})
        elif path == "/disable":
            try:
                open(LOCK_FLAG, "w").close()     # make the slicer unusable
            except OSError:
                pass
            self._json({"ok": True})
        elif path == "/save_settings":
            try:
                data = json.loads(body)
                json.dump(data, open(GUI_SETTINGS, "w"), indent=2)
                self._json({"ok": True})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
        elif path == "/slice":
            if os.path.exists(LOCK_FLAG):
                self._json({"ok": False, "error": "this tool is not for you"}, 403)
                return
            if not _busy.acquire(blocking=False):
                self._json({"ok": False, "error": "Already slicing"}, 409)
                return
            try:
                filename = urllib.parse.unquote(
                    self.headers.get("X-Filename", "model.stl"))
                settings_data = json.loads(urllib.parse.unquote(
                    self.headers.get("X-Settings", "{}")))
                run_verify = self.headers.get("X-Verify", "0") == "1"
                result = do_slice(body, filename, settings_data, run_verify)
                self._json(result)
            except MeshError as e:
                self._json({"ok": False, "error": str(e)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})
            finally:
                _busy.release()
        else:
            self.send_error(404)


def main():
    port = 8347
    # find a free port starting at the default
    for p in range(port, port + 20):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                port = p
                break
            except OSError:
                continue
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Non-Planar Spiral Slicer running at {url}")
    print("Close this window to quit.")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
