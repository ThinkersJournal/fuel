#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Remove the `mkl` feature's `extern crate` block from every `fuel-examples/examples/*/main.rs`.

The `mkl` feature of the workspace was removed (ROADMAP row 168 area, PR "drop-mkl"), so each example
that carried

    #[cfg(feature = "mkl")]
    extern crate intel_mkl_src;

loses those two lines. When the block sits between two blank lines, the blank line AFTER it goes too
(so two blank lines never become adjacent); otherwise nothing else changes. Line endings are kept per
file (a CRLF file stays CRLF). Files are enumerated from the git index (`git ls-files`), never by
walking the disk, so other worktrees are never touched.

    python scripts/drop_mkl_examples.py            # apply, print applied/population
    python scripts/drop_mkl_examples.py --check    # change nothing; exit 1 if any file still needs it

Rerun on a clean checkout of `origin/main`: the resulting `git diff` over these paths is the one
this PR carries (the PR body states the hashes).
"""

import subprocess
import sys

PATTERN = ('#[cfg(feature = "mkl")]', "extern crate intel_mkl_src;")


def tracked_mains():
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "fuel-examples/examples/*/main.rs"],
        check=True,
        capture_output=True,
    ).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def transform(data: bytes):
    """Return (new bytes, number of blocks removed). Raises if the block appears more than once."""
    eol = b"\r\n" if b"\r\n" in data else b"\n"
    lines = data.split(eol)
    hits = [
        i
        for i in range(len(lines) - 1)
        if lines[i].decode() == PATTERN[0] and lines[i + 1].decode() == PATTERN[1]
    ]
    if len(hits) > 1:
        raise SystemExit("more than one mkl block in one file; refusing")
    if not hits:
        return data, 0
    i = hits[0]
    drop = {i, i + 1}
    before_blank = i > 0 and lines[i - 1] == b""
    after_blank = i + 2 < len(lines) and lines[i + 2] == b""
    if before_blank and after_blank:
        drop.add(i + 2)
    kept = [l for k, l in enumerate(lines) if k not in drop]
    return eol.join(kept), 1


def main():
    check = "--check" in sys.argv[1:]
    files = tracked_mains()
    applied = 0
    for path in files:
        with open(path, "rb") as f:
            data = f.read()
        new, n = transform(data)
        if n:
            applied += 1
            if not check:
                with open(path, "wb") as f:
                    f.write(new)
    print(f"mkl extern-crate block: {'would remove' if check else 'removed'} from {applied} of {len(files)} main.rs files (git index)")
    if check and applied:
        sys.exit(1)


if __name__ == "__main__":
    main()
