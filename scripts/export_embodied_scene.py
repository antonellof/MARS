#!/usr/bin/env python3
"""
export_embodied_scene.py — Convert kids-ball corpus .bin to JSONL (human-readable).

Each line: node_id, modality name, episode_id, timestamp (no raw embedding).

Requires corpus from:
  ./demos/embodied_scene/bench_kids_sweep --dump-only out.bin N [D]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np

KIDS_MAGIC = 0x5344494B
MOD_NAMES = {0: "TEXT", 1: "AUDIO", 2: "IMAGE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("-o", "--output", required=True, help="Output .jsonl path")
    args = ap.parse_args()

    with open(args.corpus, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != KIDS_MAGIC:
            print(f"bad magic {magic:#x}", file=sys.stderr)
            sys.exit(1)
        struct.unpack("<I", f.read(4))
        N, D = struct.unpack("<ii", f.read(8))
        emb = np.fromfile(f, count=N * D, dtype=np.float32)
        ts = np.fromfile(f, count=N, dtype=np.float32)
        mod = np.fromfile(f, count=N, dtype=np.int32)
        epi = np.fromfile(f, count=N, dtype=np.int32)

    with open(args.output, "w") as out:
        for i in range(N):
            m = int(mod[i])
            rec = {
                "node_id": i,
                "modality": MOD_NAMES.get(m, str(m)),
                "episode_id": int(epi[i]),
                "timestamp_s": float(ts[i]),
                "embedding_dim": int(D),
            }
            out.write(json.dumps(rec) + "\n")
    print(f"Wrote {N} lines to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
