"""Immutable, process-shared lifecycle for file-mapped KT host weights.

The operating system already shares clean file-backed pages between processes.
This module provides the missing safety contract for using those pages directly
as KTransformers kernel weights:

* a content manifest identifies every checkpoint file by size and SHA-256;
* a generation binds that content to one checkpoint inode and NUMA contract;
* every attaching process owns a PID/start-time-checked lease;
* stale leases are reaped under a local file lock; and
* safetensors mappings can be changed from private-writable to read-only.

The hot attach path intentionally does not hash hundreds of gigabytes.  It
checks the strong manifest identity and file metadata.  The manifest must be
created from verified artifact metadata or once with ``build_manifest``.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Final, cast, final

_MANIFEST_SCHEMA_VERSION: Final = 1
_GENERATION_SCHEMA_VERSION: Final = 1
_LEASE_SCHEMA_VERSION: Final = 1
_MANIFEST_KIND: Final = "kt_shared_host_weights_manifest"
_GENERATION_KIND: Final = "kt_shared_host_weights_generation"
_LEASE_KIND: Final = "kt_shared_host_weights_lease"
_CONTENT_KIND: Final = "kt_shared_host_weights_content"
_MAXIMUM_JSON_BYTES: Final = 64 * 1024 * 1024
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN: Final = re.compile(r"^[A-Za-z0-9_.-]{16,256}$")
_PROT_READ: Final = 0x1
_MADV_DONTDUMP: Final = 16


class SharedHostWeightError(RuntimeError):
    """Raised when a shared-host-weight contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class SharedHostWeightFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SharedHostWeightManifest:
    content_id: str
    numa_nodes: tuple[int, ...]
    files: tuple[SharedHostWeightFile, ...]


@dataclass(frozen=True, slots=True)
class ProtectedMapping:
    start_address: int
    end_address: int
    device: int
    inode: int
    path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_fd(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    _ = os.lseek(file_descriptor, 0, os.SEEK_SET)
    while chunk := os.read(file_descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _json_object(raw: bytes, description: str) -> dict[str, object]:
    if len(raw) > _MAXIMUM_JSON_BYTES:
        raise SharedHostWeightError(f"{description} is too large")
    try:
        value = cast(object, json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SharedHostWeightError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SharedHostWeightError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _read_regular_file(path: Path, description: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(path, flags)
    except OSError as error:
        raise SharedHostWeightError(f"cannot open {description}: {path}") from error
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SharedHostWeightError(f"{description} is not a regular file: {path}")
        if before.st_size > _MAXIMUM_JSON_BYTES:
            raise SharedHostWeightError(f"{description} is too large: {path}")
        raw = bytearray()
        while chunk := os.read(
            file_descriptor, min(1024 * 1024, _MAXIMUM_JSON_BYTES + 1 - len(raw))
        ):
            raw.extend(chunk)
            if len(raw) > _MAXIMUM_JSON_BYTES:
                raise SharedHostWeightError(f"{description} is too large: {path}")
        after = os.fstat(file_descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise SharedHostWeightError(
                f"{description} changed while it was read: {path}"
            )
        return bytes(raw), after
    finally:
        os.close(file_descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SharedHostWeightError("manifest file path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise SharedHostWeightError(f"manifest file path is unsafe: {value!r}")
    normalized = path.as_posix()
    if normalized != value or "\x00" in value:
        raise SharedHostWeightError(f"manifest file path is not canonical: {value!r}")
    return value


def _validate_numa_nodes(values: object) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise SharedHostWeightError("NUMA contract must be a non-empty JSON list")
    nodes: list[int] = []
    for value in cast(list[object], values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SharedHostWeightError("NUMA node IDs must be non-negative integers")
        nodes.append(value)
    if len(nodes) != len(set(nodes)):
        raise SharedHostWeightError("NUMA node IDs must be unique")
    return tuple(nodes)


def shared_host_weight_content_id(files: Sequence[SharedHostWeightFile]) -> str:
    descriptor = {
        "files": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in files
        ],
        "kind": _CONTENT_KIND,
        "schema_version": _MANIFEST_SCHEMA_VERSION,
    }
    return _sha256_bytes(_canonical_json_bytes(descriptor))


def manifest_json(manifest: SharedHostWeightManifest) -> dict[str, object]:
    return {
        "content_id": manifest.content_id,
        "files": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in manifest.files
        ],
        "kind": _MANIFEST_KIND,
        "numa_nodes": list(manifest.numa_nodes),
        "schema_version": _MANIFEST_SCHEMA_VERSION,
    }


def _parse_manifest(raw: bytes) -> SharedHostWeightManifest:
    value = _json_object(raw, "shared-host-weight manifest")
    if (
        value.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or value.get("kind") != _MANIFEST_KIND
    ):
        raise SharedHostWeightError("unsupported shared-host-weight manifest")
    content_id = value.get("content_id")
    if not isinstance(content_id, str) or _SHA256.fullmatch(content_id) is None:
        raise SharedHostWeightError("manifest content_id is not a SHA-256 digest")
    numa_nodes = _validate_numa_nodes(value.get("numa_nodes"))
    raw_files_value = value.get("files")
    if not isinstance(raw_files_value, list) or not raw_files_value:
        raise SharedHostWeightError("manifest must contain at least one file")
    raw_files = cast(list[object], raw_files_value)
    files: list[SharedHostWeightFile] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise SharedHostWeightError("manifest file entries must be objects")
        file_object = cast(dict[str, object], raw_file)
        relative_path = _validate_relative_path(file_object.get("path"))
        size_bytes = file_object.get("size_bytes")
        sha256 = file_object.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise SharedHostWeightError(
                f"invalid size for manifest file {relative_path}"
            )
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise SharedHostWeightError(
                f"invalid SHA-256 for manifest file {relative_path}"
            )
        files.append(SharedHostWeightFile(relative_path, size_bytes, sha256))
    if tuple(sorted(entry.path for entry in files)) != tuple(
        entry.path for entry in files
    ):
        raise SharedHostWeightError("manifest files must be sorted by canonical path")
    if len({entry.path for entry in files}) != len(files):
        raise SharedHostWeightError("manifest contains duplicate file paths")
    manifest = SharedHostWeightManifest(content_id, numa_nodes, tuple(files))
    if shared_host_weight_content_id(manifest.files) != manifest.content_id:
        raise SharedHostWeightError("manifest content_id does not match its file table")
    return manifest


def load_manifest(path: Path) -> tuple[SharedHostWeightManifest, str]:
    raw, _ = _read_regular_file(path, "shared-host-weight manifest")
    return _parse_manifest(raw), _sha256_bytes(raw)


def _checkpoint_file(
    checkpoint_root: Path,
    relative_path: str,
) -> tuple[Path, int, os.stat_result]:
    root = checkpoint_root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
        _ = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise SharedHostWeightError(
            f"checkpoint file escapes its root: {relative_path}"
        ) from error
    current = root
    for part in PurePosixPath(relative_path).parts:
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise SharedHostWeightError(
                    f"checkpoint file traverses a symlink: {relative_path}"
                )
        except OSError as error:
            raise SharedHostWeightError(
                f"cannot inspect checkpoint file: {relative_path}"
            ) from error
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(resolved, flags)
    except OSError as error:
        raise SharedHostWeightError(
            f"cannot open checkpoint file: {relative_path}"
        ) from error
    metadata = os.fstat(file_descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(file_descriptor)
        raise SharedHostWeightError(
            f"checkpoint entry is not a regular file: {relative_path}"
        )
    return resolved, file_descriptor, metadata


def verify_checkpoint_files(
    checkpoint_root: Path,
    manifest: SharedHostWeightManifest,
    *,
    full_hash: bool,
) -> dict[str, os.stat_result]:
    """Verify manifest files and return stable descriptor metadata.

    ``full_hash=False`` is the intended attach path after an artifact manager
    has verified the manifest.  It never faults all model pages merely to start
    another inference process.
    """

    verified: dict[str, os.stat_result] = {}
    for entry in manifest.files:
        _, file_descriptor, before = _checkpoint_file(checkpoint_root, entry.path)
        try:
            if before.st_size != entry.size_bytes:
                raise SharedHostWeightError(
                    f"checkpoint file size changed: {entry.path}"
                )
            if full_hash and _sha256_fd(file_descriptor) != entry.sha256:
                raise SharedHostWeightError(
                    f"checkpoint file digest changed: {entry.path}"
                )
            after = os.fstat(file_descriptor)
            if _stat_identity(before) != _stat_identity(after):
                raise SharedHostWeightError(
                    f"checkpoint file changed during verification: {entry.path}"
                )
            verified[entry.path] = after
        finally:
            os.close(file_descriptor)
    return verified


def _require_effectively_read_only(
    checkpoint_root: Path,
    manifest: SharedHostWeightManifest,
    metadata: dict[str, os.stat_result],
) -> None:
    for entry in manifest.files:
        path = checkpoint_root.joinpath(*PurePosixPath(entry.path).parts)
        mount_read_only = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
        mode_read_only = metadata[entry.path].st_mode & 0o222 == 0
        if not mount_read_only and not mode_read_only:
            requirement = "checkpoint files must be on a read-only mount or have all write bits removed"
            raise SharedHostWeightError(f"{requirement} before sharing: {entry.path}")


def build_manifest(
    checkpoint_root: Path,
    *,
    numa_nodes: Sequence[int],
    relative_paths: Iterable[str] | None = None,
) -> SharedHostWeightManifest:
    """Hash a checkpoint once and return its immutable sharing manifest."""

    root = checkpoint_root.resolve(strict=True)
    if relative_paths is None:
        relative_paths = (
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    normalized_paths = sorted(
        {_validate_relative_path(path) for path in relative_paths}
    )
    if not normalized_paths:
        raise SharedHostWeightError("cannot build a manifest for an empty checkpoint")
    validated_numa = _validate_numa_nodes(list(numa_nodes))
    files: list[SharedHostWeightFile] = []
    for relative_path in normalized_paths:
        _, file_descriptor, before = _checkpoint_file(root, relative_path)
        try:
            digest = _sha256_fd(file_descriptor)
            after = os.fstat(file_descriptor)
            if _stat_identity(before) != _stat_identity(after):
                raise SharedHostWeightError(
                    f"checkpoint file changed while hashing: {relative_path}"
                )
            files.append(SharedHostWeightFile(relative_path, before.st_size, digest))
        finally:
            os.close(file_descriptor)
    frozen_files = tuple(files)
    return SharedHostWeightManifest(
        content_id=shared_host_weight_content_id(frozen_files),
        numa_nodes=validated_numa,
        files=frozen_files,
    )


def write_manifest(path: Path, manifest: SharedHostWeightManifest) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = _canonical_json_bytes(manifest_json(manifest)) + b"\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb", closefd=True) as output:
            _ = output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _process_start_time_ticks(process_id: int) -> int:
    try:
        raw = Path(f"/proc/{process_id}/stat").read_text()
        return int(raw.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError) as error:
        raise SharedHostWeightError(
            f"cannot read process identity for PID {process_id}"
        ) from error


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as error:
        raise SharedHostWeightError("cannot read Linux boot identity") from error
    if not value:
        raise SharedHostWeightError("Linux boot identity is empty")
    return value


def _process_matches(
    process_id: object, start_time_ticks: object, boot_id: object
) -> bool:
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or isinstance(start_time_ticks, bool)
        or not isinstance(start_time_ticks, int)
        or start_time_ticks <= 0
        or not isinstance(boot_id, str)
    ):
        return False
    try:
        return boot_id == _boot_id() and start_time_ticks == _process_start_time_ticks(
            process_id
        )
    except SharedHostWeightError:
        return False


def _state_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise SharedHostWeightError(
            "shared-host-weight state directory must be absolute"
        )
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SharedHostWeightError(
            "shared-host-weight state path is not a real directory"
        )
    if metadata.st_uid != os.geteuid():
        raise SharedHostWeightError(
            "shared-host-weight state directory has the wrong owner"
        )
    os.chmod(path, 0o700)
    leases = path / "leases"
    leases.mkdir(mode=0o700, exist_ok=True)
    lease_metadata = leases.lstat()
    if (
        not stat.S_ISDIR(lease_metadata.st_mode)
        or stat.S_ISLNK(lease_metadata.st_mode)
        or lease_metadata.st_uid != os.geteuid()
    ):
        raise SharedHostWeightError("shared-host-weight lease directory is unsafe")
    os.chmod(leases, 0o700)
    return path


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(path, flags, 0o600)
    try:
        payload = _canonical_json_bytes(value) + b"\n"
        _ = os.write(file_descriptor, payload)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _replace_json(path: Path, value: dict[str, object]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb", closefd=True) as output:
            _ = output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_small_json(path: Path, description: str) -> dict[str, object]:
    raw, _ = _read_regular_file(path, description)
    return _json_object(raw, description)


@final
class SharedHostWeightLease:
    """An exact, cleanup-safe attachment to one immutable weight generation."""

    def __init__(
        self,
        *,
        checkpoint_root: Path,
        manifest_path: Path,
        state_directory: Path,
        expected_content_id: str,
        expected_numa_nodes: Sequence[int],
    ) -> None:
        if _SHA256.fullmatch(expected_content_id) is None:
            raise SharedHostWeightError(
                "expected content identity is not a SHA-256 digest"
            )
        manifest, manifest_sha256 = load_manifest(manifest_path)
        expected_nodes = _validate_numa_nodes(list(expected_numa_nodes))
        if manifest.content_id != expected_content_id:
            raise SharedHostWeightError(
                "shared-host-weight content identity does not match"
            )
        if manifest.numa_nodes != expected_nodes:
            raise SharedHostWeightError(
                "shared-host-weight NUMA contract does not match"
            )
        root = checkpoint_root.resolve(strict=True)
        root_metadata = root.stat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise SharedHostWeightError("checkpoint root is not a directory")
        self.file_metadata = verify_checkpoint_files(root, manifest, full_hash=False)
        _require_effectively_read_only(root, manifest, self.file_metadata)

        state = _state_directory(state_directory)
        lock_path = state / ".lock"
        lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
        lock_metadata = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.geteuid()
        ):
            os.close(lock_descriptor)
            raise SharedHostWeightError("shared-host-weight lock file is unsafe")
        os.fchmod(lock_descriptor, 0o600)
        process_id = os.getpid()
        process_start = _process_start_time_ticks(process_id)
        boot_id = _boot_id()
        contract = {
            "checkpoint_device": root_metadata.st_dev,
            "checkpoint_inode": root_metadata.st_ino,
            "checkpoint_root": str(root),
            "content_id": manifest.content_id,
            "manifest_sha256": manifest_sha256,
            "numa_nodes": list(manifest.numa_nodes),
        }
        contract_sha256 = _sha256_bytes(_canonical_json_bytes(contract))
        lease_token = secrets.token_urlsafe(32)
        generation_path = state / "generation.json"
        leases_path = state / "leases"
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            active_leases = self._reap_stale_leases(leases_path)
            generation: dict[str, object] | None = None
            if generation_path.exists():
                generation = _load_small_json(
                    generation_path, "shared-host-weight generation"
                )
                if (
                    generation.get("schema_version") != _GENERATION_SCHEMA_VERSION
                    or generation.get("kind") != _GENERATION_KIND
                ):
                    raise SharedHostWeightError(
                        "shared-host-weight generation is invalid"
                    )
            if (
                generation is not None
                and generation.get("contract_sha256") != contract_sha256
            ):
                if active_leases:
                    raise SharedHostWeightError(
                        "another shared-host-weight generation has active leases"
                    )
                generation = None
            if generation is None:
                generation_id = str(uuid.uuid4())
                generation = {
                    "contract": contract,
                    "contract_sha256": contract_sha256,
                    "created_at_utc": _utc_now(),
                    "generation_id": generation_id,
                    "kind": _GENERATION_KIND,
                    "owner_boot_id": boot_id,
                    "owner_pid": process_id,
                    "owner_start_time_ticks": process_start,
                    "schema_version": _GENERATION_SCHEMA_VERSION,
                }
                _replace_json(generation_path, generation)
                role = "owner"
            else:
                generation_id = generation.get("generation_id")
                if (
                    not isinstance(generation_id, str)
                    or _SAFE_TOKEN.fullmatch(generation_id) is None
                ):
                    raise SharedHostWeightError(
                        "shared-host-weight generation ID is invalid"
                    )
                role = "attacher"

            lease_id = f"{process_id}.{uuid.uuid4()}"
            lease_path = leases_path / f"{generation_id}.{lease_id}.json"
            lease_record: dict[str, object] = {
                "boot_id": boot_id,
                "content_id": manifest.content_id,
                "created_at_utc": _utc_now(),
                "generation_id": generation_id,
                "kind": _LEASE_KIND,
                "lease_id": lease_id,
                "lease_token": lease_token,
                "numa_nodes": list(manifest.numa_nodes),
                "pid": process_id,
                "schema_version": _LEASE_SCHEMA_VERSION,
                "start_time_ticks": process_start,
            }
            _write_json_exclusive(lease_path, lease_record)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

        self.checkpoint_root = root
        self.manifest = manifest
        self.manifest_path = manifest_path.resolve(strict=True)
        self.manifest_sha256 = manifest_sha256
        self.state_directory = state
        self.generation_id = generation_id
        self.lease_id = lease_id
        self.role = role
        self._lease_path = lease_path
        self._lease_token = lease_token
        self._process_id = process_id
        self._process_start_time_ticks = process_start
        self._released = False
        _ = atexit.register(self.release)

    @staticmethod
    def _reap_stale_leases(leases_path: Path) -> list[dict[str, object]]:
        active: list[dict[str, object]] = []
        for lease_path in sorted(leases_path.glob("*.json")):
            try:
                lease = _load_small_json(lease_path, "shared-host-weight lease")
            except SharedHostWeightError:
                lease_path.unlink()
                continue
            if (
                lease.get("schema_version") != _LEASE_SCHEMA_VERSION
                or lease.get("kind") != _LEASE_KIND
                or not _process_matches(
                    lease.get("pid"),
                    lease.get("start_time_ticks"),
                    lease.get("boot_id"),
                )
            ):
                lease_path.unlink()
                continue
            active.append(lease)
        return active

    def require_mapped_files(self, paths: Iterable[Path]) -> None:
        manifest_paths = {entry.path for entry in self.manifest.files}
        for path in paths:
            try:
                relative = (
                    path.resolve(strict=True)
                    .relative_to(self.checkpoint_root)
                    .as_posix()
                )
            except (OSError, ValueError) as error:
                raise SharedHostWeightError(
                    f"mapped weight is outside checkpoint: {path}"
                ) from error
            if relative not in manifest_paths:
                raise SharedHostWeightError(
                    f"mapped weight is absent from content manifest: {relative}"
                )
            expected = self.file_metadata[relative]
            actual = path.stat()
            if _stat_identity(expected) != _stat_identity(actual):
                raise SharedHostWeightError(
                    f"mapped weight changed before attach: {relative}"
                )

    def release(self) -> None:
        if self._released:
            return
        if (
            os.getpid() != self._process_id
            or _process_start_time_ticks(os.getpid()) != self._process_start_time_ticks
        ):
            # A forked child inherits Python objects and atexit registrations,
            # but it does not own the parent's lease record.
            return
        lock_path = self.state_directory / ".lock"
        lock_flags = os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        try:
            lock_descriptor = os.open(lock_path, lock_flags)
        except OSError:
            return
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            if self._lease_path.exists():
                lease = _load_small_json(self._lease_path, "shared-host-weight lease")
                if (
                    lease.get("generation_id") != self.generation_id
                    or lease.get("lease_id") != self.lease_id
                    or lease.get("lease_token") != self._lease_token
                ):
                    raise SharedHostWeightError(
                        "refusing to remove a foreign weight lease"
                    )
                self._lease_path.unlink()
            active = self._reap_stale_leases(self.state_directory / "leases")
            generation_path = self.state_directory / "generation.json"
            if not active and generation_path.exists():
                generation = _load_small_json(
                    generation_path, "shared-host-weight generation"
                )
                if generation.get("generation_id") == self.generation_id:
                    generation_path.unlink()
            self._released = True
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    def __enter__(self) -> SharedHostWeightLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _mapping_lines() -> Iterable[tuple[int, int, str, int, int, str]]:
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError as error:
        raise SharedHostWeightError("cannot inspect process memory mappings") from error
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            continue
        address_range, permissions, _, device_text, inode_text = fields[:5]
        path = fields[5] if len(fields) == 6 else ""
        try:
            start_text, end_text = address_range.split("-", 1)
            major_text, minor_text = device_text.split(":", 1)
            yield (
                int(start_text, 16),
                int(end_text, 16),
                permissions,
                os.makedev(int(major_text, 16), int(minor_text, 16)),
                int(inode_text),
                path,
            )
        except ValueError:
            continue


def protect_file_mappings_read_only(
    paths: Iterable[Path],
    *,
    dont_dump: bool = True,
) -> tuple[ProtectedMapping, ...]:
    """Apply ``PROT_READ`` to every current mapping of the exact input inodes."""

    expected: dict[tuple[int, int], str] = {}
    for path in paths:
        metadata = path.stat()
        expected[(metadata.st_dev, metadata.st_ino)] = str(path.resolve(strict=True))
    if not expected:
        raise SharedHostWeightError("no mapped files were supplied")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    protected: list[ProtectedMapping] = []
    covered: set[tuple[int, int]] = set()
    for start, end, permissions, device, inode, mapped_path in _mapping_lines():
        identity = (device, inode)
        if identity not in expected:
            continue
        if "x" in permissions or "s" in permissions:
            raise SharedHostWeightError(
                f"unexpected executable/shared mapping for host weight: {mapped_path}"
            )
        length = end - start
        if libc.mprotect(ctypes.c_void_p(start), length, _PROT_READ) != 0:
            error_number = ctypes.get_errno()
            raise SharedHostWeightError(
                f"mprotect(PROT_READ) failed for {mapped_path}: {os.strerror(error_number)}"
            )
        if (
            dont_dump
            and libc.madvise(ctypes.c_void_p(start), length, _MADV_DONTDUMP) != 0
        ):
            error_number = ctypes.get_errno()
            raise SharedHostWeightError(
                f"madvise(MADV_DONTDUMP) failed for {mapped_path}: {os.strerror(error_number)}"
            )
        protected.append(
            ProtectedMapping(start, end, device, inode, expected[identity])
        )
        covered.add(identity)
    missing = set(expected) - covered
    if missing:
        missing_paths = ", ".join(expected[identity] for identity in sorted(missing))
        raise SharedHostWeightError(
            f"files have no current memory mapping: {missing_paths}"
        )
    return tuple(protected)


def _parse_numa_cli(value: str) -> tuple[int, ...]:
    try:
        return _validate_numa_nodes([int(item) for item in value.split(",") if item])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "NUMA nodes must be comma-separated integers"
        ) from error


@final
class _Arguments(argparse.Namespace):
    command: str = ""
    checkpoint_root: Path | None = None
    output: Path | None = None
    numa_nodes: tuple[int, ...] | None = None
    manifest: Path | None = None
    full_hash: bool = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest")
    _ = build.add_argument("--checkpoint-root", type=Path, required=True)
    _ = build.add_argument("--output", type=Path, required=True)
    _ = build.add_argument("--numa-nodes", type=_parse_numa_cli, required=True)
    verify = subparsers.add_parser("verify-manifest")
    _ = verify.add_argument("--checkpoint-root", type=Path, required=True)
    _ = verify.add_argument("--manifest", type=Path, required=True)
    _ = verify.add_argument("--full-hash", action="store_true")
    arguments = _Arguments()
    _ = parser.parse_args(namespace=arguments)
    if arguments.checkpoint_root is None:
        raise SharedHostWeightError("checkpoint root argument is missing")
    if arguments.command == "build-manifest":
        if arguments.output is None or arguments.numa_nodes is None:
            raise SharedHostWeightError("manifest output or NUMA nodes are missing")
        output = arguments.output.resolve()
        root = arguments.checkpoint_root.resolve(strict=True)
        relative_paths = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink() and path.resolve() != output
        ]
        manifest = build_manifest(
            root,
            numa_nodes=arguments.numa_nodes,
            relative_paths=relative_paths,
        )
        write_manifest(arguments.output, manifest)
        print(json.dumps(manifest_json(manifest), indent=2, sort_keys=True))
        return 0
    if arguments.manifest is None:
        raise SharedHostWeightError("manifest argument is missing")
    manifest, manifest_sha256 = load_manifest(arguments.manifest)
    _ = verify_checkpoint_files(
        arguments.checkpoint_root,
        manifest,
        full_hash=arguments.full_hash,
    )
    print(
        json.dumps(
            {
                "content_id": manifest.content_id,
                "files": len(manifest.files),
                "manifest_sha256": manifest_sha256,
                "verified_full_hash": arguments.full_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
