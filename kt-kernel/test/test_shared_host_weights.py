"""CPU-only tests for immutable shared-host-weight lifecycle."""

from __future__ import annotations

import importlib.util
import json
import mmap
import os
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "python" / "utils" / "shared_host_weights.py"
SPEC = importlib.util.spec_from_file_location("shared_host_weights_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
shared = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shared
SPEC.loader.exec_module(shared)


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model-00001.safetensors").write_bytes(b"first-weight-file")
    (checkpoint / "model-00002.safetensors").write_bytes(b"second-weight-file")
    (checkpoint / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    return checkpoint


def _manifest(tmp_path: Path, checkpoint: Path):
    manifest = shared.build_manifest(checkpoint, numa_nodes=(0, 1))
    for entry in manifest.files:
        (checkpoint / entry.path).chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    shared.write_manifest(manifest_path, manifest)
    return manifest, manifest_path


def test_manifest_is_content_addressed_and_detects_tampering(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest, manifest_path = _manifest(tmp_path, checkpoint)

    loaded, _ = shared.load_manifest(manifest_path)
    assert loaded == manifest
    assert shared.shared_host_weight_content_id(loaded.files) == loaded.content_id
    shared.verify_checkpoint_files(checkpoint, loaded, full_hash=True)

    (checkpoint / "model-00001.safetensors").write_bytes(b"tampered-weight!!")
    with pytest.raises(shared.SharedHostWeightError, match="digest changed"):
        shared.verify_checkpoint_files(checkpoint, loaded, full_hash=True)


def test_generation_supports_multiple_leases_and_exact_numa_contract(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest, manifest_path = _manifest(tmp_path, checkpoint)
    state = tmp_path / "state"

    first = shared.SharedHostWeightLease(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=state,
        expected_content_id=manifest.content_id,
        expected_numa_nodes=(0, 1),
    )
    second = shared.SharedHostWeightLease(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=state,
        expected_content_id=manifest.content_id,
        expected_numa_nodes=(0, 1),
    )
    assert first.role == "owner"
    assert second.role == "attacher"
    assert first.generation_id == second.generation_id
    assert len(tuple((state / "leases").glob("*.json"))) == 2

    with pytest.raises(shared.SharedHostWeightError, match="NUMA contract"):
        shared.SharedHostWeightLease(
            checkpoint_root=checkpoint,
            manifest_path=manifest_path,
            state_directory=state,
            expected_content_id=manifest.content_id,
            expected_numa_nodes=(0,),
        )

    first.release()
    assert (state / "generation.json").exists()
    second.release()
    assert not (state / "generation.json").exists()
    assert not tuple((state / "leases").glob("*.json"))


def test_stale_lease_is_reaped_before_new_generation(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest, manifest_path = _manifest(tmp_path, checkpoint)
    state = tmp_path / "state"

    first = shared.SharedHostWeightLease(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=state,
        expected_content_id=manifest.content_id,
        expected_numa_nodes=(0, 1),
    )
    old_generation = first.generation_id
    first.release()

    leases = state / "leases"
    stale_path = leases / "stale.json"
    stale_path.write_text(
        json.dumps(
            {
                "boot_id": shared._boot_id(),
                "generation_id": old_generation,
                "kind": "kt_shared_host_weights_lease",
                "pid": 2**30,
                "schema_version": 1,
                "start_time_ticks": 1,
            }
        )
    )

    second = shared.SharedHostWeightLease(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=state,
        expected_content_id=manifest.content_id,
        expected_numa_nodes=(0, 1),
    )
    assert not stale_path.exists()
    second.release()


def test_require_mapped_files_rejects_unmanifested_file(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest, manifest_path = _manifest(tmp_path, checkpoint)
    outside = checkpoint / "unmanifested.safetensors"
    outside.write_bytes(b"not-in-manifest")
    lease = shared.SharedHostWeightLease(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=tmp_path / "state",
        expected_content_id=manifest.content_id,
        expected_numa_nodes=(0, 1),
    )
    with pytest.raises(shared.SharedHostWeightError, match="absent from content manifest"):
        lease.require_mapped_files((outside,))
    lease.release()


def test_writable_checkpoint_is_rejected_for_read_only_sharing(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest = shared.build_manifest(checkpoint, numa_nodes=(0, 1))
    manifest_path = tmp_path / "manifest.json"
    shared.write_manifest(manifest_path, manifest)
    with pytest.raises(shared.SharedHostWeightError, match="write bits removed"):
        shared.SharedHostWeightLease(
            checkpoint_root=checkpoint,
            manifest_path=manifest_path,
            state_directory=tmp_path / "state",
            expected_content_id=manifest.content_id,
            expected_numa_nodes=(0, 1),
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork ownership check")
def test_forked_child_cannot_release_parent_lease(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest, manifest_path = _manifest(tmp_path, checkpoint)
    lease = shared.SharedHostWeightLease(
        checkpoint_root=checkpoint,
        manifest_path=manifest_path,
        state_directory=tmp_path / "state",
        expected_content_id=manifest.content_id,
        expected_numa_nodes=(0, 1),
    )
    child = os.fork()
    if child == 0:
        lease.release()
        os._exit(0)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert lease._lease_path.exists()
    lease.release()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and mprotect contract")
def test_file_mapping_is_made_read_only_without_copy(tmp_path: Path) -> None:
    source = tmp_path / "mapped.safetensors"
    source.write_bytes(os.urandom(8192))
    with source.open("r+b") as source_file:
        mapping = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            protected = shared.protect_file_mappings_read_only((source,))
            assert protected
            matching_permissions = []
            address = protected[0].start_address
            for line in Path("/proc/self/maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                start_text, end_text = fields[0].split("-")
                if int(start_text, 16) <= address < int(end_text, 16):
                    matching_permissions.append(fields[1])
            assert matching_permissions
            assert all("w" not in permissions for permissions in matching_permissions)
            assert mapping[0] == source.read_bytes()[0]
        finally:
            mapping.close()
