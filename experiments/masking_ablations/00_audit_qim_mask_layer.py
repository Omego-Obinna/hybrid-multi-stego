#!/usr/bin/env python3
"""Audit whether the current Ours-QIM code exposes a formal mask stream."""
from __future__ import annotations
import argparse, csv, json, os, re
from pathlib import Path
MASK_TERMS=[r"\bmask_bits?\b",r"\bmu_bits?\b",r"\bmasked_payload\b",r"\bb_prime\b",r"\bunmask\b",r"F_mask",r"F_unmask"]
COST_TERMS=[r"qim_base_cost",r"qim_modulated_cost",r"qim_texture_alpha",r"qim_dither_alpha",r"rho_p1",r"rho_m1"]
SKIP={".git","__pycache__","venv",".venv","node_modules","shared_srm_cache"}

def scan(path:Path):
    try:text=path.read_text(encoding="utf-8",errors="replace")
    except Exception:return [],[]
    return ([p for p in MASK_TERMS if re.search(p,text,re.I)], [p for p in COST_TERMS if re.search(p,text,re.I)])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",required=True); ap.add_argument("--out-dir",required=True); a=ap.parse_args()
    root=Path(a.project_root).expanduser().resolve(); out=Path(a.out_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for current,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('.')]
        for name in files:
            p=Path(current)/name
            if p.suffix.lower() not in {'.py','.csv','.json','.jsonl','.tex','.txt'}:continue
            try:
                if p.stat().st_size>10*1024*1024:continue
            except OSError:continue
            mh,ch=scan(p)
            if mh or ch:rows.append({'path':str(p),'mask_layer_hits':'; '.join(mh),'cost_modulation_hits':'; '.join(ch)})
    with (out/'qim_mask_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['path','mask_layer_hits','cost_modulation_hits']); w.writeheader(); w.writerows(rows)
    mask_files=[r for r in rows if r['mask_layer_hits']]; cost_files=[r for r in rows if r['cost_modulation_hits']]
    summary={'project_root':str(root),'files_with_mask_layer_terms':len(mask_files),'files_with_cost_modulation_terms':len(cost_files),'interpretation':('Mask-layer terms were found; inspect them before equating the current image-cost QIM with the formal F_mask construction.' if mask_files else 'No explicit mask-layer bitstream was found by keyword scan. Treat the ablation as an operational instantiation of the formal QIM mask layer, separate from image-cost modulation.')}
    (out/'qim_mask_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('[OK] Audit CSV:',out/'qim_mask_audit.csv'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
