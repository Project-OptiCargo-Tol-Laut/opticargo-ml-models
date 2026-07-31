from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


class ArtifactCompatibilityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    return {
        "python": __import__("platform").python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def save_bundle(
    path: Path,
    estimator: Any,
    feature_columns: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete_metadata = {**metadata, "dependencies": dependency_versions()}
    bundle = {
        "estimator": estimator,
        "feature_columns": feature_columns,
        "metadata": complete_metadata,
    }
    joblib.dump(bundle, path, compress=3)
    checksum = sha256_file(path)
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps({**complete_metadata, "sha256": checksum}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "artifact_path": str(path),
        "metadata_path": str(metadata_path),
        "sha256": checksum,
        "metadata": complete_metadata,
    }


def _check_sidecar(path: Path) -> None:
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_checksum = metadata.get("sha256")
    if expected_checksum and sha256_file(path) != expected_checksum:
        raise ArtifactCompatibilityError("Checksum artifact tidak cocok dengan metadata sidecar.")
    trained = metadata.get("dependencies", {}).get("scikit_learn")
    if trained and trained != sklearn.__version__:
        raise ArtifactCompatibilityError(
            "Versi scikit-learn artifact tidak kompatibel: "
            f"trained={trained}, runtime={sklearn.__version__}. Jalankan training ulang pada environment runtime."
        )


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    _check_sidecar(path)
    bundle = joblib.load(path)
    required = {"estimator", "feature_columns", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Artifact model tidak mempunyai struktur bundle yang valid.")
    return bundle


def _s3_client(endpoint: str, secure: bool, access_key: str, secret_key: str):
    import boto3

    scheme = "https" if secure else "http"
    endpoint_url = endpoint if endpoint.startswith(("http://", "https://")) else f"{scheme}://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def upload_to_minio(
    path: Path,
    *,
    endpoint: str,
    secure: bool,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_key: str,
) -> str:
    if not access_key or not secret_key:
        raise ValueError("Credential MinIO belum dikonfigurasi.")
    client = _s3_client(endpoint, secure, access_key, secret_key)
    client.upload_file(str(path), bucket, object_key)
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.exists():
        client.upload_file(str(metadata_path), bucket, f"{object_key}.metadata.json")
    return f"s3://{bucket}/{object_key}"


def download_from_minio(
    destination: Path,
    *,
    endpoint: str,
    secure: bool,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_key: str,
) -> Path:
    if not access_key or not secret_key:
        raise ValueError("Credential MinIO belum dikonfigurasi.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = _s3_client(endpoint, secure, access_key, secret_key)
    client.download_file(bucket, object_key, str(destination))
    try:
        client.download_file(bucket, f"{object_key}.metadata.json", str(destination.with_suffix(".metadata.json")))
    except Exception:
        pass
    return destination
