"""
Entry point of the portable Windows package.

Starts the Streamlit app from the bundled Python and opens it in the default
browser once the server is up. Folder layout of the package:

    PenguinDetector/
        START.bat        double-click to run
        launcher.py      this file
        python/          bundled Python with all libraries
        app/             the Streamlit app with model weights
"""
import os
import socket
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT, "app")

# Work fully offline: never wait on network checks or model hub downloads
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("YOLO_OFFLINE", "True")
# Keep Ultralytics settings inside the package instead of the user profile
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, "settings"))


def _free_port(start: int = 8501) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found between 8501 and 8550")


def _open_browser_when_ready(url: str, port: int) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    if os.environ.get("PENGUIN_NO_BROWSER") != "1":
        webbrowser.open(url)
    print()
    print("=" * 60)
    print("  Penguin Detector працює.")
    print(f"  Якщо браузер не відкрився сам, відкрийте адресу: {url}")
    print()
    print("  НЕ ЗАКРИВАЙТЕ це вікно, поки працюєте з програмою.")
    print("  Щоб завершити роботу, просто закрийте це вікно.")
    print("=" * 60)
    print()


def main() -> int:
    os.chdir(APP_DIR)
    sys.path.insert(0, APP_DIR)

    port = _free_port()
    url = f"http://localhost:{port}"
    print("Запуск Penguin Detector... Зачекайте, будь ласка (до хвилини).")
    threading.Thread(target=_open_browser_when_ready, args=(url, port), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", os.path.join(APP_DIR, "app.py"),
        "--server.headless=true",          # no first-run e-mail prompt in the console
        "--server.address=localhost",      # reachable only from this computer
        f"--server.port={port}",
        "--server.fileWatcherType=none",   # code never changes in the package
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
