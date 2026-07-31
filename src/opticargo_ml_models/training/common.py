from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path, manifest: dict[str, Any]) -> dict[str, str]:
    checksum = sha256_file(output)
    payload = {
        **manifest,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": checksum,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": str(output), "manifest": str(manifest_path), "sha256": checksum}
