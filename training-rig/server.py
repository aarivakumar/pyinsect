#!/usr/bin/env python3
"""Offline training rig: a phone takes insect photos, this computer saves and sorts them.

Run:  python3 training-rig/server.py          (phone on a USB cable, using adb)
      python3 training-rig/server.py --lan    (phone on USB tethering or your own hotspot)

It also shows the Pyinsect site at http://localhost:8000. Needs only Python 3.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

RIG = Path(__file__).resolve().parent
SITE = RIG.parent
DATA = RIG / "data"                      # never put in git
RUNTIME = SITE / "assets" / "vendor" / "ort"
CDN = b"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"
SPLITS = ("feature", "validation")
MAX_BYTES = 10 * 1024 * 1024
LOCK = threading.Lock()                  # one change to the folders at a time


class RigError(Exception):
    """A problem to show the person using the rig."""


def clean_label(name):
    """An insect name that is also safe to use as a folder name."""
    name = " ".join(name.split())
    if not re.fullmatch(r"\w[\w \-]{0,39}", name):
        raise RigError("Use letters, numbers, spaces or - in the name (up to 40).")
    return name


def find_label(name):
    """Use the folder we already have, so 'ladybug' and 'Ladybug' stay together."""
    name = clean_label(name)
    return next((n for n in labels() if n.lower() == name.lower()), name)


def photo_path(query):
    split, file = query.get("split", ""), query.get("file", "")
    if split not in SPLITS or not re.fullmatch(r"[\w\-]+\.jpg", file):
        raise RigError("That is not a photo.")
    path = DATA / clean_label(query.get("label", "")) / split / file
    if not path.is_file():
        raise RigError("That photo is gone.")
    return path


def labels():
    return sorted((p.name for p in DATA.iterdir() if p.is_dir()), key=str.lower) if DATA.is_dir() else []


def photos(label, split):
    return sorted(p.name for p in (DATA / label / split).glob("*.jpg"))


def current_label():
    try:
        return (DATA / "current.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def set_label(name):
    label = find_label(name)
    for split in SPLITS:
        (DATA / label / split).mkdir(parents=True, exist_ok=True)
    (DATA / "current.txt").write_text(label, encoding="utf-8")
    return label


def write_csv(label):
    with open(DATA / label / "labels.csv", "w", newline="", encoding="utf-8") as f:
        rows = csv.writer(f)
        rows.writerow(["file", "split", "label"])
        for split in SPLITS:
            rows.writerows([name, split, label] for name in photos(label, split))


def save_photo(name, data):
    if not data.startswith(b"\xff\xd8\xff"):
        raise RigError("Only JPEG photos can be saved.")
    label = find_label(name)
    for split in SPLITS:
        (DATA / label / split).mkdir(parents=True, exist_ok=True)
    held, kept = len(photos(label, "validation")), len(photos(label, "feature"))
    # Every 5th photo is held back to test the AI
    split = "validation" if (held + 1) * 5 <= held + kept + 1 else "feature"
    file = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**9:09d}.jpg"
    (DATA / label / split / file).write_bytes(data)
    write_csv(label)
    return {"label": label, "split": split, "file": file}


def overview():
    return {"current": current_label(),
            "labels": [{"label": n, **{s: photos(n, s) for s in SPLITS}} for n in labels()]}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream"}
    lan = False
    timeout = 30                         # drop a connection that stops sending

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def log_message(self, format, *args):
        pass                             # keep the terminal easy to read

    def send_bytes(self, data, kind, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, value, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(value).encode(), "application/json", status)

    def allowed(self):
        """Stop other websites open in a browser from reading or changing our photos."""
        host = self.headers.get("Host", "")
        if not self.lan and urlsplit("//" + host).hostname not in ("localhost", "127.0.0.1"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin == "http://" + host

    def do_GET(self):
        if not self.allowed():
            return self.send_error(HTTPStatus.FORBIDDEN)
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/ping":
            return self.send_json({"ok": True, "label": current_label()})
        if url.path == "/api/overview":
            with LOCK:
                return self.send_json(overview())
        if url.path == "/photo":
            try:
                return self.send_bytes(photo_path(query).read_bytes(), "image/jpeg")
            except (RigError, OSError):
                return self.send_error(HTTPStatus.NOT_FOUND)
        if url.path in ("/capture", "/rig"):
            page = RIG / "web" / (url.path[1:] + ".html")
            return self.send_bytes(page.read_bytes(), "text/html; charset=utf-8")
        return self.send_site()

    do_HEAD = do_GET

    def send_site(self):
        """The Pyinsect site, without hidden files or the rig's own folder."""
        path = Path(self.translate_path(self.path)).resolve()
        if path != SITE and SITE not in path.parents:
            return self.send_error(HTTPStatus.NOT_FOUND)
        parts = path.relative_to(SITE).parts
        if any(p.startswith(".") for p in parts) or (parts and parts[0].lower() == RIG.name):
            return self.send_error(HTTPStatus.NOT_FOUND)
        if path.is_dir():
            url = urlsplit(self.path)
            if not url.path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", "/" + url.path.strip("/") + "/")
                self.send_header("Content-Length", "0")
                return self.end_headers()
            path = path / "index.html"
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        if path.suffix == ".html":
            page = path.read_bytes()
            if (RUNTIME / "ort.wasm.min.js").is_file():
                # Offline: use our saved copy of the AI engine instead of the internet one
                page = page.replace(CDN, b"/assets/vendor/ort/ort.wasm.min.js")
            return self.send_bytes(page, "text/html; charset=utf-8")
        with open(path, "rb") as f:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            if self.command != "HEAD":
                shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        if not self.allowed():
            return self.send_error(HTTPStatus.FORBIDDEN)
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/upload":
                size = int(self.headers.get("Content-Length") or 0)
                if not 0 < size <= MAX_BYTES:
                    raise RigError("The photo is empty or too big.")
                data = self.rfile.read(size)
                if len(data) != size:    # the cable came out halfway
                    self.close_connection = True
                    return
                with LOCK:
                    saved = save_photo(query.get("label", ""), data)
                print(f"Saved {saved['label']} photo ({saved['split']})")
                return self.send_json(saved)
            with LOCK:
                if url.path == "/api/label":
                    return self.send_json({"label": set_label(query.get("name", ""))})
                if url.path in ("/api/move", "/api/delete"):
                    path = photo_path(query)
                    label = path.parent.parent.name
                    if url.path == "/api/delete":
                        path.unlink()
                    else:
                        other = SPLITS[1 - SPLITS.index(path.parent.name)]
                        (DATA / label / other).mkdir(exist_ok=True)
                        path.replace(DATA / label / other / path.name)
                    write_csv(label)
                    return self.send_json({"ok": True})
            return self.send_error(HTTPStatus.NOT_FOUND)
        except RigError as error:
            return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except (OSError, ValueError) as error:
            print(f"Problem: {unquote(url.path)}: {error}")
            # Not a 400: the phone keeps the photo and tries again later
            self.close_connection = True


def connect_phone(port):
    """Let the phone open http://localhost:<port> through the USB cable."""
    if not shutil.which("adb"):
        print("adb is not installed, so a USB phone can't connect yet (see README.md).")
        return
    try:
        done = subprocess.run(["adb", "reverse", f"tcp:{port}", f"tcp:{port}"],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"adb did not answer: {error}")
        return
    if done.returncode == 0:
        print(f"USB phone connected. On the phone open http://localhost:{port}/capture")
    else:
        print("No USB phone yet. Plug it in, tap Allow, then start this again.")
        print("  adb said: " + done.stderr.strip())


def main():
    options = argparse.ArgumentParser(description="Offline training rig")
    options.add_argument("--port", type=int, default=8000)
    options.add_argument("--lan", action="store_true",
                         help="let other devices on this network open the rig")
    args = options.parse_args()
    DATA.mkdir(exist_ok=True)
    Handler.lan = args.lan
    try:
        server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), Handler)
    except OSError as error:
        raise SystemExit(f"Port {args.port} is busy. Close the other server, or add --port 8001. ({error})")
    print(f"Computer page: http://localhost:{args.port}/rig")
    if not (RUNTIME / "ort-wasm-simd-threaded.wasm").is_file():
        print("The AI will need the internet. To fix: python3 training-rig/get_runtime.py")
    if args.lan:
        print(f"--lan: anyone on this network can open it. Phone: http://<this computer's address>:{args.port}/capture")
    else:
        connect_phone(args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
