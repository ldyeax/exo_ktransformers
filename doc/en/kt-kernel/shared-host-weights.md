# Shared resident AMXINT4 host weights

KTransformers normally opens converted safetensors with a private file mapping,
allocates one anonymous `BufferB` for every projection/expert/NUMA partition,
and copies each packed weight and scale into that allocation. The source file
pages can be shared by Linux, but the `BufferB` copies cannot.

For the GLM-5.2 checkpoint at `/mnt/sanic/glm52-AMXINT4`, the safetensor
headers describe 405,000,724,992 bytes of packed weights and 1,275,068,416
bytes of scales: 378.374 GiB copied into private anonymous memory by every
model process.

The opt-in shared path binds AMXINT4 `BufferB` directly to safetensors tensor
addresses. It keeps the mappings and Python tensor views alive, changes every
mapping to `PROT_READ`, marks them `MADV_DONTDUMP`, and skips all private
weight allocation and copying. Separate prefill and decode processes map the
same checkpoint inodes, so Linux accounts the clean physical pages once.

Safetensors guarantees 8-byte tensor alignment rather than the private
allocator's former 64-byte alignment. The AMXINT4 AVX and AMX read paths use
unaligned AVX-512 loads for packed weights and scales; a focused kernel test
runs both paths from read-only mappings at exactly 8 mod 64. `BufferB` is a
non-owning view and its destructor never frees or unmaps those addresses.

This is only enabled for preconverted, merged AMXINT4 safetensors. Online
quantization, AMXINT8, and the per-expert `.kt` file layout continue to use the
copying loader and fail closed if forced into this mode.

## Prepare an immutable manifest

The manifest contains a SHA-256 for every checkpoint file. Creating it by
hashing the checkpoint is a one-time operation; artifact deployment may write
the same schema from its already-verified file hashes instead.

```bash
kt-shared-host-weights build-manifest \
  --checkpoint-root /mnt/sanic/glm52-AMXINT4 \
  --output /var/lib/exo/manifests/glm52-amxint4-shared.json \
  --numa-nodes 0,1
```

The files must be on a read-only mount or have all write permission bits
removed before a process can attach. This prevents an in-place checkpoint
write from changing clean pages behind a running kernel.

An optional one-time end-to-end verification is:

```bash
kt-shared-host-weights verify-manifest \
  --checkpoint-root /mnt/sanic/glm52-AMXINT4 \
  --manifest /var/lib/exo/manifests/glm52-amxint4-shared.json \
  --full-hash
```

The hot attach path checks the strong manifest identity, file sizes/inodes, and
read-only status without rereading 378 GiB.

## Launch

Set these variables identically in every prefill/decode process on one host:

```bash
export KT_SHARED_HOST_WEIGHTS=1
export KT_SHARED_HOST_WEIGHTS_MANIFEST=/var/lib/exo/manifests/glm52-amxint4-shared.json
export KT_SHARED_HOST_WEIGHTS_CONTENT_ID=<content_id printed by build-manifest>
export KT_SHARED_HOST_WEIGHTS_STATE_DIR=/var/lib/exo/shared-host-weights/glm52
```

Continue to pass the converted directory through `--kt-weight-path`. The NUMA
contract comes from the actual KT worker-pool mapping and must exactly match
the manifest. For the dwagon TP=2 process it is `0,1`; a mismatched order or
count is rejected.

The first process owns a random generation under the local state directory.
Every process creates an owner-only lease containing its PID, Linux start
time, boot ID, content identity, and generation. Attachers must match the
checkpoint root inode, manifest digest, content ID, and ordered NUMA nodes.
Stale leases are reaped under `flock`; the generation record is removed only
after the final live lease releases. A forked child cannot release its
parent's lease.

The state directory contains lifecycle metadata only. Removing its final
generation does not remove checkpoint files, and process exit naturally
unmaps the file-backed pages.

## NUMA behavior

Converted tensors retain their `.numa.0` / `.numa.1` split. Opening a mapping,
changing its protection, and obtaining a data pointer do not fault the tensor
contents. The matching KT NUMA worker first-touches its own tensor pages during
inference. Avoid hashing or explicitly prefetching the checkpoint from an
unbound process immediately before launch, because that can place page-cache
pages on the wrong socket.

Use `Pss`, `Private_Clean`, `Private_Dirty`, and `Shared_Clean` from
`/proc/<pid>/smaps_rollup` when validating two processes. RSS counts a shared
file page in each process and therefore overstates physical use; the expected
regression signal is removal of roughly 378.374 GiB of anonymous/private
`BufferB` storage per additional GLM-5.2 process.
