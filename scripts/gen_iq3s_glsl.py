#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate fuel-kernels-source/kernels/qmatvec_iq3_s.glsl from its template and `IQ3S_GRID`.

The 512-entry grid in the shader is never typed: it is read from `fuel-quantized/src/iq_grids.rs`
(itself generated from llama.cpp's ggml-common.h at 9d57ce456c by scripts/gen_iq_grids.py).

  python scripts/gen_iq3s_glsl.py write   # rewrite the .glsl
  python scripts/gen_iq3s_glsl.py check   # exit 1 if the committed .glsl differs from the template + grid

Then rebuild the SPIR-V: `bash fuel-kernels-source/kernels/compile.sh qmatvec_iq3_s.glsl`.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRIDS = ROOT / "fuel-quantized/src/iq_grids.rs"
TEMPLATE = ROOT / "fuel-kernels-source/kernels/templates/qmatvec_iq3_s.glsl.in"
OUT = ROOT / "fuel-kernels-source/kernels/qmatvec_iq3_s.glsl"


def grid():
    text = GRIDS.read_text(encoding="utf-8")
    m = re.search(r"pub const IQ3S_GRID: \[u32; 512\] = \[(.*?)\];", text, re.S)
    assert m, "IQ3S_GRID not found"
    vals = re.findall(r"0x([0-9a-fA-F]{8})", m.group(1))
    assert len(vals) == 512, f"expected 512 entries, parsed {len(vals)}"
    return vals


def render():
    vals = grid()
    rows = []
    for i in range(0, 512, 8):
        rows.append("    " + ", ".join(f"0x{v}u" for v in vals[i : i + 8]))
    body = "const uint IQ3S_GRID[512] = uint[512](\n" + ",\n".join(rows) + "\n);"
    t = TEMPLATE.read_text(encoding="utf-8")
    assert t.count("@IQ3S_GRID@") == 1
    return t.replace("@IQ3S_GRID@", body)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    out = render()
    if mode == "write":
        OUT.write_text(out, encoding="utf-8", newline="\n")
        print(f"wrote {OUT.relative_to(ROOT)}: 512 grid entries")
    elif mode == "check":
        cur = OUT.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
        if cur != out:
            print("qmatvec_iq3_s.glsl differs from template + IQ3S_GRID; run `write`")
            sys.exit(1)
        print("qmatvec_iq3_s.glsl matches template + IQ3S_GRID (512 entries)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
