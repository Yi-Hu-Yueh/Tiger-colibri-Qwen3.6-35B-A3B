import hashlib
import zipfile

import pytest

from app.services.colibri_installer import parse_checksums, validate_zip_members, verify_checksum
from app.services.github_release import select_release_assets


def test_release_filters_official_assets_only():
    assets = [
        {"name": "colibri-v1.2.3-windows-x86_64.zip", "browser_download_url": "https://evil.example/file.zip", "size": 1},
        {"name": "colibri-v1.2.3-windows-x86_64.zip", "browser_download_url": "https://github.com/JustVugg/colibri/releases/download/v1.2.3/colibri-v1.2.3-windows-x86_64.zip", "size": 2},
        {"name": "SHA256SUMS.txt", "browser_download_url": "https://github.com/JustVugg/colibri/releases/download/v1.2.3/SHA256SUMS.txt", "size": 3},
    ]
    selected = select_release_assets(assets)
    assert selected["windows_asset"]["size"] == 2
    assert selected["checksum_asset"]["name"] == "SHA256SUMS.txt"


def test_checksum_parse_and_verify(tmp_path):
    archive = tmp_path / "file.zip"
    archive.write_bytes(b"official")
    digest = hashlib.sha256(b"official").hexdigest()
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(f"{digest}  file.zip\n", encoding="utf-8")
    assert parse_checksums(sums.read_text())["file.zip"] == digest
    assert verify_checksum(archive, sums)["status"] == "PASS"
    archive.write_bytes(b"tampered")
    assert verify_checksum(archive, sums)["status"] == "FAIL"


@pytest.mark.parametrize("member", ["../evil.exe", "/absolute/evil.exe", "C:/evil.exe", "ok/../../evil.exe"])
def test_zip_slip_rejected(tmp_path, member):
    with pytest.raises(ValueError):
        validate_zip_members([member], tmp_path / "runtime")


def test_safe_zip_member_allowed(tmp_path):
    validate_zip_members(["colibri/bin/qwen36.exe"], tmp_path / "runtime")
