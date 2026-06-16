from __future__ import annotations

import argparse
import json
import mimetypes
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .api import (
    read_result_metadata,
    read_review_items,
    run_auto_correction,
    submit_manual_correction,
)


class M11ApiHandler(BaseHTTPRequestHandler):
    server_version = "GeoCoreM11API/0.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/m1-1/health":
                self._send_json({"status": "ok", "module": "M1-1"})
                return
            if parsed.path == "/api/m1-1/results":
                query = parse_qs(parsed.query)
                output_dir = _single_query_value(query, "output_dir", "outputs/m1_1")
                self._send_json(read_result_metadata(output_dir))
                return
            if parsed.path == "/api/m1-1/review-items":
                query = parse_qs(parsed.query)
                output_dir = _single_query_value(query, "output_dir", "outputs/m1_1")
                only = _single_query_value(query, "only_needs_review", "true").lower() != "false"
                self._send_json(read_review_items(output_dir, only_needs_review=only))
                return
            if parsed.path == "/api/m1-1/file":
                query = parse_qs(parsed.query)
                file_path = _single_query_value(query, "path", "")
                self._send_file(file_path)
                return
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")
        except Exception as exc:
            self._send_exception(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/m1-1/run":
                self._send_json(run_auto_correction(self._read_json()))
                return
            if parsed.path == "/api/m1-1/manual-correction":
                self._send_json(submit_manual_correction(self._read_json()))
                return
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")
        except Exception as exc:
            self._send_exception(exc)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: str) -> None:
        if not file_path:
            self._send_error(HTTPStatus.BAD_REQUEST, "Query parameter 'path' is required.")
            return
        path = Path(unquote(file_path)).resolve()
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"status": "error", "error": message}, status=status)

    def _send_exception(self, exc: Exception) -> None:
        traceback.print_exc()
        status = HTTPStatus.BAD_REQUEST if isinstance(exc, (ValueError, FileNotFoundError)) else HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json(
            {
                "status": "error",
                "error": str(exc),
                "exception_type": exc.__class__.__name__,
            },
            status=status,
        )

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")


def run_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), M11ApiHandler)
    print(f"M1-1 API server listening on http://{host}:{port}")
    print("Health check: GET /api/m1-1/health")
    server.serve_forever()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1-1 backend API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_server(host=args.host, port=args.port)
    return 0


def _single_query_value(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    if not values:
        return default
    return values[0]


if __name__ == "__main__":
    raise SystemExit(main())
