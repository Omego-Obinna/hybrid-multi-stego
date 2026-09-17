#!/usr/bin/env python3
"""Export packed binary sequences to ASCII for the NIST STS file-input path."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--config-id')
    ap.add_argument('--payload', type=float)
    ap.add_argument('--message-class', default='zero')
    ap.add_argument('--max-files', type=int, default=30)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.config_id:
        manifest = manifest[manifest['config_id'] == args.config_id]
    if args.payload is not None:
        manifest = manifest[np.isclose(manifest['payload_bpp'], args.payload)]
    manifest = manifest[manifest['message_class'] == args.message_class].head(args.max_files)

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in manifest.iterrows():
        source = Path(row['path'])
        packed = np.frombuffer(source.read_bytes(), dtype=np.uint8)
        bits = np.unpackbits(packed, bitorder='big')[: int(row['actual_bits'])]
        target = out / f'{source.stem}.txt'
        target.write_text(''.join('1' if bit else '0' for bit in bits), encoding='ascii')
        rows.append({**row.to_dict(), 'ascii_path': str(target)})
        print(f'[OK] {target} ({len(bits)} bits)')

    pd.DataFrame(rows).to_csv(out / 'nist_ascii_manifest.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
