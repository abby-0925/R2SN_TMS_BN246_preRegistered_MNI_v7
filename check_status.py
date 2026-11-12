#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TMS BN246-v6 cohort status and ROI missingness summary."""
import argparse,csv,json
from collections import Counter
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/00_manifest/manifest_TMS_ses01.csv')
    p.add_argument('--out-root',default='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects')
    p.add_argument('--tag',default='TMS_BN246_T1prep_v6')
    a=p.parse_args()
    with open(a.manifest,newline='',encoding='utf-8-sig') as f:subjects=list(csv.DictReader(f))
    root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sub in subjects:
        sid=sub['id'];sf=root/sid/a.tag/'qc'/'run_summary.json'
        try:
            info=json.loads(sf.read_text(encoding='utf-8'))
            state=info.get('status','UNKNOWN')
            net=root/sid/a.tag/'network'/f'{sid}_network.npy'
            if state in ('SUCCESS','SUCCESS_WITH_MISSING_NODES') and not net.is_file():
                state='MISSING_NETWORK_FILE'
            row=dict(id=sid,code=sub.get('code',sid),lesion_mode=sub.get('lesion_mode',''),group=sub.get('group',''),status=state,
                     n_valid=info.get('n_valid',''),n_zero=info.get('n_zero',''),
                     n_other_missing=info.get('n_other_missing',''),
                     n_lesioned_rois=info.get('n_lesioned_rois',''),
                     error=info.get('error',''))
        except FileNotFoundError:
            row=dict(id=sid,code=sub.get('code',sid),lesion_mode=sub.get('lesion_mode',''),group=sub.get('group',''),status='NOT_STARTED',n_valid='',n_zero='',
                     n_other_missing='',n_lesioned_rois='',error='')
        except Exception as exc:
            row=dict(id=sid,code=sub.get('code',sid),lesion_mode=sub.get('lesion_mode',''),group=sub.get('group',''),status='READ_ERROR',n_valid='',n_zero='',
                     n_other_missing='',n_lesioned_rois='',error=str(exc))
        rows.append(row)
    path=root/f'{a.tag}_cohort_status.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id','code','status']);wr.writeheader();wr.writerows(rows)
    print('Total:',len(rows))
    print('Counts:',dict(Counter(r['status'] for r in rows)))
    print('CSV:',path)

if __name__=='__main__':main()
