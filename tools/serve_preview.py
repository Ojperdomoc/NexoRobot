#!/usr/bin/env python3
"""Servidor local para ver la app de inspección.

Sirve el repositorio por HTTP en 0.0.0.0 (para que funcione detrás de la vista
previa o en otro equipo de la red) y redirige la raíz a `/inspeccion/`.

Uso:
    python3 tools/serve_preview.py            # http://localhost:8080/inspeccion/
    PORT=9000 python3 tools/serve_preview.py  # otro puerto

No cachea: al recargar siempre se ven los cambios. La cámara del navegador
exige HTTPS o `localhost`; en producción la app se sirve desde GitHub Pages.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    """Archivos estáticos del repo + redirección de la raíz a la app."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".onnx": "application/octet-stream",
        ".webmanifest": "application/manifest+json",
    }

    def _redirect_root(self) -> bool:
        if self.path.split("?")[0] in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/inspeccion/")
            self.end_headers()
            return True
        return False

    def do_GET(self):  # noqa: N802 (API de http.server)
        if not self._redirect_root():
            super().do_GET()

    def do_HEAD(self):  # noqa: N802 (API de http.server)
        if not self._redirect_root():
            super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), partial(Handler, directory=str(ROOT)))
    print(f"NexoRobot · inspección en http://0.0.0.0:{port}/inspeccion/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
