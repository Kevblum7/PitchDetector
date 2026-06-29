"""Unit tests for file checksums / duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.app.services.checksum import sha256_file


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    data = b"pitch tip detector test bytes" * 100
    f = tmp_path / "a.bin"
    f.write_bytes(data)
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_identical_content_same_digest(tmp_path: Path) -> None:
    data = b"identical"
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(data)
    b.write_bytes(data)
    assert sha256_file(a) == sha256_file(b)


def test_different_content_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert sha256_file(a) != sha256_file(b)
