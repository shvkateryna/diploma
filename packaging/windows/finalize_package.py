"""
Turn the CI-built zip into the package handed to users: adds the model weights
(not in the repo) and the PDF guide.

    python packaging/windows/finalize_package.py PenguinDetector-windows.zip ІНСТРУКЦІЯ.pdf

Writes PenguinDetector-windows-final.zip next to the input zip.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"


def main() -> None:
    ci_zip, guide = Path(sys.argv[1]), Path(sys.argv[2])
    out = ci_zip.with_name(ci_zip.stem + "-final.zip")

    config = yaml.safe_load((APP / "config.yaml").read_text(encoding="utf-8"))
    weights = [APP / m["path"] for m in config["models"].values()]
    missing = [str(w) for w in weights if not w.exists()]
    if missing:
        sys.exit(f"Missing model weights: {missing}")

    shutil.copyfile(ci_zip, out)
    with zipfile.ZipFile(out, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        names = set(zf.namelist())
        for w in weights:
            arc = f"PenguinDetector/app/{w.relative_to(APP).as_posix()}"
            if arc in names:
                sys.exit(f"{arc} is already in the zip")
            zf.write(w, arc)
            print("added", arc)
        zf.write(guide, f"PenguinDetector/{guide.name}")
        print("added", f"PenguinDetector/{guide.name}")

    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
