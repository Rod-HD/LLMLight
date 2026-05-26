"""Serve the LLMLight side-by-side viewer with a clickable URL banner.

Usage:
    python scripts/serve_viewer.py            # port 8000
    python scripts/serve_viewer.py 8123       # custom port

Prints an OSC-8 hyperlink (clickable in Windows Terminal / VS Code / iTerm2)
plus the plain URL as fallback for terminals that don't support OSC-8.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import sys
from pathlib import Path


def _osc8(url: str, label: str) -> str:
    """Wrap text as an OSC-8 hyperlink escape sequence."""
    esc = "\x1b]8;;"
    bel = "\x1b\\"
    return f"{esc}{url}{bel}{label}{esc}{bel}"


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = Path(__file__).resolve().parent.parent
    handler = http.server.SimpleHTTPRequestHandler

    # ThreadingTCPServer so big replay files don't block the manifest fetch.
    class ReuseTCP(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    import os
    os.chdir(root)

    with ReuseTCP(("", port), handler) as httpd:
        local = f"http://127.0.0.1:{port}/viewer/"
        lan = f"http://{_local_ip()}:{port}/viewer/"
        bar = "=" * 64
        print(bar)
        print(f"  LLMLight viewer is live")
        print(f"  Local : {_osc8(local, local)}")
        print(f"  LAN   : {_osc8(lan, lan)}")
        root_str = str(root).encode("ascii", "replace").decode("ascii")
        print(f"  Root  : {root_str}")
        print(f"  Press Ctrl+C to stop")
        print(bar, flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[stop] viewer server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
