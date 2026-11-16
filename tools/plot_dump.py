#!/usr/bin/env python3
"""Decode and plot a matrix written by `mblas-bench --dump_matrix`.

Standalone: needs numpy, and matplotlib only if you want the PNG.
Reads <file>.bin and its description <file>.bin.json (written next to it).

  plot_dump.py mblas_dump_a.bin                # stats + mblas_dump_a.png
  plot_dump.py mblas_dump_a.bin --no-plot      # stats only
  plot_dump.py a.bin b.bin --out cmp.png       # overlay several dumps in one figure
"""
import argparse
import json
import re
import sys

import numpy as np


def minifloat_lut(nbits, ebits, mbits, bias, signed=True, ieee_special=False, nan_codes=(),
                  fnuz=False):
    """Value of every code of a small float format (OCP / FNUZ style)."""
    lut = np.zeros(1 << nbits, dtype=np.float64)
    emask = (1 << ebits) - 1
    for code in range(1 << nbits):
        sign = (code >> (nbits - 1)) & 1 if signed else 0
        e = (code >> mbits) & emask
        m = code & ((1 << mbits) - 1)
        if e == 0:
            v = (m / (1 << mbits)) * 2.0 ** (1 - bias)
        else:
            v = (1 + m / (1 << mbits)) * 2.0 ** (e - bias)
        if ieee_special and e == emask:
            v = np.inf if m == 0 else np.nan
        if code in nan_codes:
            v = np.nan
        lut[code] = -v if sign else v
    if fnuz:
        lut[1 << (nbits - 1)] = np.nan  # the only NaN; there is no negative zero
    return lut


# name fragment -> (bits per code, decoder). Order matters: FNUZ before plain.
def _formats():
    f = {}
    f["8F_E4M3_FNUZ"] = (8, minifloat_lut(8, 4, 3, 8, fnuz=True))
    f["8F_E5M2_FNUZ"] = (8, minifloat_lut(8, 5, 2, 16, fnuz=True))
    f["8F_UE4M3"] = (8, minifloat_lut(8, 4, 3, 7, signed=False, nan_codes=(0x7F, 0xFF)))
    f["8F_UE5M3"] = (8, minifloat_lut(8, 5, 3, 15, signed=False))
    f["8F_UE8M0"] = (8, np.array([2.0 ** (c - 127) if c != 255 else np.nan for c in range(256)]))
    f["8F_E4M3"] = (8, minifloat_lut(8, 4, 3, 7, nan_codes=(0x7F, 0xFF)))
    f["8F_E5M2"] = (8, minifloat_lut(8, 5, 2, 15, ieee_special=True))
    f["6F_E2M3"] = (6, minifloat_lut(6, 2, 3, 1))
    f["6F_E3M2"] = (6, minifloat_lut(6, 3, 2, 3))
    f["4F_E2M1"] = (4, minifloat_lut(4, 2, 1, 1))
    return f


MINIFLOATS = _formats()
PLAIN = {  # name fragment -> numpy dtype (or the special string 'bf16')
    "R_64F": np.float64, "R_32F": np.float32, "R_16F": np.float16, "R_16BF": "bf16",
    "R_8I": np.int8, "R_8U": np.uint8, "R_16I": np.int16, "R_16U": np.uint16,
    "R_32I": np.int32, "R_32U": np.uint32,
}


def unpack_codes(raw, bits, packing, n):
    """Raw device bytes -> one integer code per element (fp4: low nibble is first)."""
    b = np.frombuffer(raw, dtype=np.uint8)
    if packing == 2:
        codes = np.empty(b.size * 2, dtype=np.uint8)
        codes[0::2] = b & 0xF
        codes[1::2] = b >> 4
        return codes[:n]
    if bits == 6:
        return (b & 0x3F)[:n]
    return b[:n]


def decode(meta, raw):
    """Return (values float64, codes uint array or None, bits per code)."""
    tname = meta["type"]
    n = meta["num_elements"]
    for frag, (bits, lut) in MINIFLOATS.items():
        if frag in tname:
            codes = unpack_codes(raw, bits, meta["packing"], n)
            return lut[codes], codes, bits
    for frag, dt in PLAIN.items():
        if tname.endswith(frag) or f"_{frag}" in tname:
            if dt == "bf16":
                u = np.frombuffer(raw, dtype=np.uint16)[:n].astype(np.uint32) << 16
                vals = u.view(np.float32).astype(np.float64)
                return vals, np.frombuffer(raw, dtype=np.uint16)[:n], 16
            arr = np.frombuffer(raw, dtype=dt)[:n]
            ucode = np.frombuffer(raw, dtype=f"u{arr.itemsize}")[:n]
            return arr.astype(np.float64), ucode, arr.itemsize * 8
    raise SystemExit(f"Unsupported type in sidecar: {tname}")


def valid_view(flat, meta):
    """Drop ld padding: returns (batch, cols, rows) array in memory order."""
    rows, cols, ld = meta["rows_mem"], meta["cols_mem"], meta["ld"]
    batch, stride = meta["batch"], meta["stride"]
    need = (batch - 1) * stride + ld * cols
    if flat.size < need:
        flat = np.concatenate([flat, np.zeros(need - flat.size, dtype=flat.dtype)])
    out = np.empty((batch, cols, rows), dtype=flat.dtype)
    for b in range(batch):
        out[b] = flat[b * stride: b * stride + ld * cols].reshape(cols, ld)[:, :rows]
    return out


def popcount(a):
    a = a.astype(np.uint64)
    c = np.zeros(a.shape, dtype=np.uint8)
    while a.any():
        c += (a & 1).astype(np.uint8)
        a >>= 1
    return c


def analyse(path):
    with open(path + ".json") as f:
        meta = json.load(f)
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != meta["bytes"]:
        print(f"warning: {path} is {len(raw)} bytes, sidecar says {meta['bytes']}", file=sys.stderr)
    vals, codes, bits = decode(meta, raw)
    vv = valid_view(vals, meta)
    cv = valid_view(codes, meta)
    finite = vv[np.isfinite(vv)]
    stats = {
        "file": path, "type": meta["type"], "stage": meta.get("stage"), "init": meta.get("init"),
        "elements": int(vv.size),
        "zero_fraction": float((vv == 0).mean()),
        "nan_inf_fraction": float((~np.isfinite(vv)).mean()),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "bits_per_code": bits,
        "mean_popcount": float(popcount(cv).mean()),
        # bit flips between neighbours in memory order (down a column)
        "mean_toggles": float(popcount(cv[:, :, 1:] ^ cv[:, :, :-1]).mean()) if vv.shape[2] > 1 else 0.0,
    }
    return meta, vv, cv, bits, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin", nargs="+", help="dump .bin file(s); the .bin.json must be next to each")
    ap.add_argument("--out", help="output PNG (default: <first bin without .bin>.png)")
    ap.add_argument("--no-plot", action="store_true", help="print stats only")
    ap.add_argument("--bins", type=int, default=100, help="histogram bins for wide types")
    args = ap.parse_args()

    results = [analyse(p) for p in args.bin]
    for _, _, _, _, s in results:
        print(json.dumps(s, indent=2))
    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_v, ax_c) = plt.subplots(1, 2, figsize=(13, 4.5))
    for meta, vv, cv, bits, s in results:
        label = f"{meta['matrix'].upper()} {meta['type'].split('_', 1)[-1]} ({s['zero_fraction']:.1%} zero)"
        finite = vv[np.isfinite(vv)]
        uniq = np.unique(finite)
        if uniq.size <= 64:  # small formats: one bar per representable value
            counts = np.array([(finite == u).sum() for u in uniq]) / finite.size
            xs = np.arange(uniq.size)
            ax_v.bar(xs + 0.0, counts, alpha=0.6, label=label)
            ax_v.set_xticks(xs)
            ax_v.set_xticklabels([f"{u:g}" for u in uniq], rotation=90, fontsize=7)
            ax_v.set_xlabel("representable value")
        else:
            ax_v.hist(finite, bins=args.bins, alpha=0.6, density=True, label=label)
            ax_v.set_xlabel("value")
        ax_v.set_ylabel("fraction (small formats) / density (wide formats)")
        if bits <= 8:
            cnt = np.bincount(cv.ravel().astype(np.int64), minlength=1 << bits) / cv.size
            ax_c.bar(np.arange(cnt.size), cnt, alpha=0.6, label=label)
            ax_c.set_xlabel("raw code (hex)")
            if cnt.size <= 16:
                ax_c.set_xticks(np.arange(cnt.size))
                ax_c.set_xticklabels([f"{i:X}" for i in range(cnt.size)])
        else:
            ax_c.hist(np.log2(np.abs(finite[finite != 0])), bins=args.bins, alpha=0.6, label=label)
            ax_c.set_xlabel("log2 |value|")
    ax_v.set_title("decoded values")
    ax_c.set_title("code usage" if all(r[3] <= 8 for r in results) else "magnitude")
    ax_v.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or (args.bin[0][:-4] if args.bin[0].endswith(".bin") else args.bin[0]) + ".png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
