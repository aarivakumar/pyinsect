#!/usr/bin/env python3
"""Download the AI engine (ONNX Runtime Web) so the training rig works with no internet.

Run once while you have internet:  python3 training-rig/get_runtime.py
The files are saved in assets/vendor/ort/ and are never put in git.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/"
FOLDER = Path(__file__).resolve().parent.parent / "assets" / "vendor" / "ort"

# A fingerprint for each file, so a broken or changed download is not used
FILES = {
    "ort.wasm.min.js": "65e09376df69107e881b5c34d2d37aed333a366b6d941073ec518168e269b87d",
    "ort-wasm-simd-threaded.mjs": "30dd851d9c00622940500f71ddd2ff8820c5cb65270816080175b958705385a8",
    "ort-wasm-simd-threaded.wasm": "71aef04959c5c1b6de461b6538e2058e306610034a85aad2742d0c7fd4533fe4",
}


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = FOLDER / name
        if path.exists() and fingerprint(path.read_bytes()) == expected:
            print(f"Already have {name}")
            continue
        print(f"Downloading {name} ...")
        try:
            with urllib.request.urlopen(URL + name, timeout=60) as reply:
                data = reply.read()
        except OSError as error:
            sys.exit(f"Could not download it. Check your internet connection. ({error})")
        if fingerprint(data) != expected:
            sys.exit(f"{name} is not the file we expected, so it was not saved.")
        path.write_bytes(data)
    print("Done. The training rig can now run the AI with no internet.")


if __name__ == "__main__":
    main()
