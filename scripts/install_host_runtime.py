#!/usr/bin/env python3
"""Install the pinned PRoot path mapper into the active conda environment."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import urllib.request

URL = "https://proot.gitlab.io/proot/bin/proot"
# The upstream binary at this stable URL was rebuilt; keep the digest pinned
# so installation remains integrity checked rather than accepting arbitrary
# downloads.
SHA256 = "3f48a11a7ae3bfdc63a61f9ccc309cda4c4e833bdc14c479e7201fc35f6f312a"


def main() -> None:
    if Path(os.environ.get("CONDA_PREFIX", "/nonexistent")).resolve() != Path(sys.prefix).resolve():
        raise SystemExit("Activate the intended conda environment before installation")
    target = Path(sys.prefix) / "bin/proot"
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print(f"Verified PRoot: {target}")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("PRoot download differs from the recorded version; review the new build before updating its hash")
    temporary = target.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.chmod(0o755)
    temporary.replace(target)
    print(f"Installed PRoot: {target}")


if __name__ == "__main__":
    main()
