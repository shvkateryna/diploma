"""
Smoke test for the portable Windows package, run by CI with the bundled Python:

    PenguinDetector\\python\\python.exe smoke_test.py <package dir>

Expects placeholder weights at app/models/{yolo_aug.pt,rfdetr.pth}. Checks that
both models run through the real app flow and that the launcher serves the app.
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image

PKG = os.path.abspath(sys.argv[1])
APP = os.path.join(PKG, "app")
sys.path.insert(0, APP)

failures = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


# 1. Full app flow for every configured model, on JPG and TIFF input
from streamlit.testing.v1 import AppTest  # noqa: E402

rng = np.random.default_rng(0)
img = Image.fromarray(rng.integers(0, 255, (2100, 3100, 3), dtype=np.uint8))
for model in ("yolo_augmentation", "rfdetr_baseline"):
    for ext in (".jpg", ".tif"):
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        img.save(tmp.name)
        at = AppTest.from_file(os.path.join(APP, "app.py"), default_timeout=900)
        at.run()
        at.session_state["job"] = {"path": tmp.name, "name": "test" + ext, "model": model, "conf": 0.25}
        t = time.time()
        at.run()
        res = at.session_state["results"] if "results" in at.session_state else None
        errors = [e.value for e in at.exception] + [e.value for e in at.error]
        check(f"app flow {model} {ext}", res is not None and not errors,
              f"({time.time() - t:.0f}s, errors={errors})")

# 2. Launcher starts the server and it answers
env = dict(os.environ, PENGUIN_NO_BROWSER="1", PYTHONUTF8="1")
proc = subprocess.Popen([sys.executable, os.path.join(PKG, "launcher.py")], cwd=PKG, env=env)
healthy = False
for _ in range(240):
    try:
        with urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=2) as r:
            healthy = r.read().strip() == b"ok"
        if healthy:
            break
    except OSError:
        time.sleep(0.5)
proc.kill()
check("launcher serves the app", healthy)

# 3. Paths must fit into Windows MAX_PATH (260): leave ~80 chars for the folder
#    the user extracts into (e.g. C:\Users\<name>\Downloads\PenguinDetector-windows\)
files = [os.path.join(d, f) for d, _, fs in os.walk(PKG) for f in fs]
print(f"package has {len(files)} files", flush=True)
longest = max(
    (os.path.join(d, f) for d, _, fs in os.walk(PKG) for f in fs),
    key=len,
)
rel = len(os.path.relpath(longest, PKG))
check("longest relative path", rel < 180, f"{rel} chars: {os.path.relpath(longest, PKG)}")

if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("All smoke tests passed")
