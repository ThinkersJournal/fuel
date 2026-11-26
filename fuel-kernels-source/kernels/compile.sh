#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Compile every shader source (*.wgsl, *.glsl) in this
# directory to SPIR-V, writing the output to ../../fuel-vulkan-kernels/spv/.
#
# Toolchain: requires the Vulkan SDK (for glslangValidator)
# and naga-cli (cargo install naga-cli) on PATH.
#
# SPIR-V files are committed to the repo — contributors who don't
# touch shader sources never need any of these tools installed.
#
# Usage:
#   cd fuel-graph-executor/src/shaders
#   ./compile.sh            # compile all
#   ./compile.sh rope.wgsl  # compile a single file
#
# Conventions:
#   .wgsl   -> naga (Vulkan 1.1 / SPIR-V 1.3)
#   .glsl   -> glslangValidator, compute stage, Vulkan 1.1
#
# The Slang sources are NOT here any more: since 0.46.0 they live in the
# published `fishstringer-kernels-source` crate, and the committed
# `fuel-vulkan-kernels/spv` outputs for them are regenerated from a pinned
# version of that crate (see `fuel-vulkan-kernels/spv/SOURCE.txt`).
#
# GLSL files that need a specific entry or non-default SPIR-V version
# can override via a sidecar comment line `// compile: <flags>` at
# the top of the file.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
# Outputs land in the fuel-vulkan-kernels crate's spv/ directory.
# Layout: fuel-kernels-source/kernels/  →  fuel-vulkan-kernels/spv/
OUT_DIR="$(cd "$SRC_DIR/../.." && pwd)/fuel-vulkan-kernels/spv"
mkdir -p "$OUT_DIR"

# Resolve tools. glslangValidator comes from the Vulkan SDK.
GLSLANG="${GLSLANG:-${VULKAN_SDK:-}/Bin/glslangValidator}"
NAGA="${NAGA:-naga}"

# On non-Windows the SDK usually puts binaries in bin/ (lowercase).
if [[ ! -x "$GLSLANG" && -n "${VULKAN_SDK:-}" ]]; then
  [[ -x "$VULKAN_SDK/bin/glslangValidator" ]] && GLSLANG="$VULKAN_SDK/bin/glslangValidator"
fi

check_tool() {
  if ! command -v "$1" >/dev/null 2>&1 && [[ ! -x "$1" ]]; then
    echo "error: $2 not found (looked for '$1')" >&2
    echo "       Install the Vulkan SDK and/or run 'cargo install naga-cli'." >&2
    return 1
  fi
}

compile_wgsl() {
  local src="$1"
  local base; base="$(basename "$src" .wgsl)"
  local out="$OUT_DIR/$base.spv"
  echo "  wgsl  $base.wgsl -> ../fuel-vulkan-kernels/spv/$base.spv"
  "$NAGA" "$src" "$out"
}

# The flags a GLSL file is compiled with: the default target (Vulkan 1.1), or the sidecar `// compile: <flags>`
# within the first 8 lines REPLACING it (the sidecar carries the target flags only; `-V -S comp` are always given).
glsl_target_flags() {
  local src="$1" side
  side="$(head -n 8 "$src" | tr -d '\r' | sed -n 's#^// compile:[[:space:]]*\(--target-env[[:space:]]*[A-Za-z0-9._]*\).*#\1#p' | head -n 1)"
  if [[ -n "$side" ]]; then echo "$side"; else echo "--target-env vulkan1.1"; fi
}

# Where compile_glsl writes: OUT_DIR normally, CHECK_DIR under --check.
compile_glsl() {
  local src="$1"
  local base; base="$(basename "$src" .glsl)"
  local dest="${COMPILE_OUT_DIR:-$OUT_DIR}"
  local out="$dest/$base.spv"
  local flags; flags="$(glsl_target_flags "$src")"
  echo "  glsl  $base.glsl [$flags] -> $out"
  # shellcheck disable=SC2086
  "$GLSLANG" -V -S comp $flags "$src" -o "$out" >/dev/null
}

compile_one() {
  local f="$1"
  case "$f" in
    *.wgsl)  compile_wgsl  "$f" ;;
    *.glsl)  compile_glsl  "$f" ;;
    *) echo "warning: unknown shader type: $f" >&2 ;;
  esac
}

# --check [files...]: rebuild into a temp dir and compare byte-for-byte with the committed .spv. Exits 1 if any
# file differs or is missing, listing each (a binary nobody can rebuild is an unverifiable artifact: GAP-407).
if [[ "${1:-}" == "--check" ]]; then
  shift
  check_tool "$GLSLANG" "glslangValidator"
  COMPILE_OUT_DIR="$(mktemp -d)"
  cd "$SRC_DIR"
  files=("$@")
  if (( ${#files[@]} == 0 )); then shopt -s nullglob; files=(*.glsl); fi
  ok=0; bad=0
  for f in "${files[@]}"; do
    base="$(basename "$f" .glsl)"
    compile_glsl "$f" >/dev/null
    if [[ ! -f "$OUT_DIR/$base.spv" ]]; then echo "MISSING  $base.spv (no committed file)"; bad=$((bad+1))
    elif cmp -s "$COMPILE_OUT_DIR/$base.spv" "$OUT_DIR/$base.spv"; then echo "OK       $base.spv"; ok=$((ok+1))
    else echo "DIFFERS  $base.spv  ($(glsl_target_flags "$f"))"; bad=$((bad+1)); fi
  done
  echo "glslangValidator: $("$GLSLANG" --version | head -n 1)"
  echo "check: $ok reproduced, $bad not, of $((ok+bad)) GLSL files"
  [[ $bad -eq 0 ]]
  exit $?
fi

if (( $# > 0 )); then
  # Check only the tools we actually need.
  for arg in "$@"; do
    case "$arg" in
      *.wgsl)  check_tool "$NAGA" "naga-cli" ;;
      *.glsl)  check_tool "$GLSLANG" "glslangValidator" ;;
    esac
  done
  cd "$SRC_DIR"
  for f in "$@"; do compile_one "$f"; done
else
  # Compile everything. Need both tools.
  check_tool "$NAGA" "naga-cli" || true
  check_tool "$GLSLANG" "glslangValidator" || true
  cd "$SRC_DIR"
  shopt -s nullglob
  for f in *.wgsl; do compile_one "$f"; done
  for f in *.glsl; do compile_one "$f"; done
fi

echo "Done. Output: $OUT_DIR"
