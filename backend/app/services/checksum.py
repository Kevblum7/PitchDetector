"""File checksums for duplicate detection (CLAUDE.md §9)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    Streaming keeps memory flat for large video files.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
