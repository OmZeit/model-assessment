from __future__ import annotations

import argparse
import sys


def main() -> int:
    try:
        from .ui import run_server
    except ImportError:
        try:
            from model_assessment.ui import run_server
        except ImportError as exc:
            print("Dash is not installed. Install it with: pip install dash", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 2

    parser = argparse.ArgumentParser(description="Start the local AssayReady Dash UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the Dash server.")
    parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    parser.add_argument("--allow-remote", action="store_true", help="Deprecated: remote binding is currently unsupported.")
    parser.add_argument("--server.address", dest="legacy_host", help=argparse.SUPPRESS)
    parser.add_argument("--server.port", dest="legacy_port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    host = args.legacy_host or args.host
    port = args.legacy_port or args.port
    try:
        run_server(host=host, port=port, debug=bool(args.debug), allow_remote=bool(args.allow_remote))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
