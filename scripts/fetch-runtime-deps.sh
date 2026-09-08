#!/usr/bin/env bash
# Downloads and verifies the ONNX model weights and onnxruntime shared
# library the vision package needs at runtime (see README's "Runtime
# prerequisites"). Linux x86_64, CPU-only -- for another platform or a
# CUDA-capable build, follow the README's manual steps instead.
#
# Usage: scripts/fetch-runtime-deps.sh
# Populates models/ and onnxruntime-gpu/ at the repo root (both gitignored).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
ORT_DIR="$ROOT/onnxruntime-gpu"

# Pin: matches onnxruntime_go v1.24.0's targeted onnxruntime C API version
# (see that module's own README, "Note on onnxruntime Library Versions").
# Check there and bump this if go.mod's onnxruntime_go version changes.
ORT_VERSION="1.22.0"
ORT_ASSET="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ASSET}"

HF_REPO="https://huggingface.co/fal/AuraFace-v1/resolve/main"
declare -A MODEL_SHA256=(
  [scrfd_10g_bnkps.onnx]="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
  [glintr100.onnx]="a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"
)

mkdir -p "$MODELS_DIR" "$ORT_DIR"

for name in "${!MODEL_SHA256[@]}"; do
  dest="$MODELS_DIR/$name"
  if [ -f "$dest" ] && echo "${MODEL_SHA256[$name]}  $dest" | sha256sum -c - >/dev/null 2>&1; then
    echo "already present: $name"
    continue
  fi
  echo "downloading $name ..."
  curl -sL -o "$dest" "$HF_REPO/$name"
  echo "${MODEL_SHA256[$name]}  $dest" | sha256sum -c -
done

if [ -f "$ORT_DIR/libonnxruntime.so" ]; then
  echo "already present: onnxruntime-gpu/libonnxruntime.so"
else
  echo "downloading onnxruntime ${ORT_VERSION} (CPU build) ..."
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -sL -o "$tmp/$ORT_ASSET" "$ORT_URL"
  tar xzf "$tmp/$ORT_ASSET" -C "$tmp"
  # The unversioned .so is a symlink into the same extracted tree; ship the
  # real, versioned file so it survives being copied out on its own.
  cp "$tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" \
    "$ORT_DIR/libonnxruntime.so"
fi

echo "done: $MODELS_DIR and $ORT_DIR are ready"
