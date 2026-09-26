#!/usr/bin/env python3
"""Preserve actual reference images, including older mapped-cache receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference-work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    state = json.loads((a.root / "matrix.json").read_text())
    if len(state["cells"]) != 56 or any(r["status"] in ("PENDING", "RUNNING") for r in state["cells"]):
        raise ValueError("archive only the complete frozen attempt inventory")
    a.out.mkdir()
    entries = []
    roots = (a.root.resolve(), (a.reference_work / "cache").resolve())
    for row in state["cells"]:
        attempts = [row] + [r["record"] for r in row.get("prior_attempts", [])]
        for attempt in attempts:
            directory = a.root / attempt["directory"]
            receipt_path = directory / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            images = receipt.get("reference_binaries", [])
            if attempt["status"] == "PASS" and not images:
                raise ValueError("complete capture lacks binary identities")
            for image in images:
                source = Path(image["path"]).resolve(strict=True)
                if not any(source.is_relative_to(root) for root in roots):
                    raise ValueError("unexpected reference image source")
                if sha(source) != image["sha256"]:
                    raise ValueError("reference image changed after capture")
                destination = a.out / (image["sha256"] + source.suffix)
                if not destination.exists():
                    shutil.copyfile(source, destination)
                if sha(destination) != image["sha256"]:
                    raise ValueError("reference image archive copy differs")
                entries.append(dict(capture=attempt["directory"], source=str(source),
                    receipt_sha256=sha(receipt_path), image_sha256=image["sha256"],
                    archived=destination.name, capture_status=attempt["status"],
                    superseded=attempt is not row))
    result = dict(status="ALL_RECORDED_IMAGES_HASH_VERIFIED", registered_attempts=56,
                  image_receipts=len(entries), unique_images=len({r["archived"] for r in entries}),
                  matrix_sha256=sha(a.root / "matrix.json"), entries=entries)
    (a.out / "images.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "entries"})


if __name__ == "__main__":
    main()
