#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-parallel, resumable launcher for BN246 T1prep v5 on 4090 Linux server (no Slurm)."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter

VERSION='TMS_BN246_native_T1prep_brainmask_v6.1'
ROOT='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects'
TAG='TMS_BN246_T1prep_v6'
MANIFEST='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/00_manifest/manifest_TMS_ses01.csv'


def manifest_rows(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        if not row.get('id') or not row.get('t1'):
            raise ValueError('Manifest requires non-empty id and t1 columns')
        row.setdefault('code',row.get('id',''))
        mode=(row.get('lesion_mode') or '').strip()
        if not mode:
            mode='MNI_MASK' if row.get('lesion') else 'NO_LESION'
        if mode not in ('MNI_MASK','NO_LESION'):
            raise ValueError(f"{row['id']}: lesion_mode={mode}; resolve REVIEW_REQUIRED before running")
        row['lesion_mode']=mode
        row['lesion']=(row.get('lesion') or '').strip()
        if mode=='MNI_MASK' and not row['lesion']:
            raise ValueError(f"{row['id']}: MNI_MASK requires a lesion path")
        if mode=='NO_LESION' and row['lesion']:
            raise ValueError(f"{row['id']}: NO_LESION must have an empty lesion path")
        if not Path(row['t1']).is_file():
            raise FileNotFoundError(f"{row['id']}: T1 not found: {row['t1']}")
        if mode=='MNI_MASK' and not Path(row['lesion']).is_file():
            raise FileNotFoundError(f"{row['id']}: MNI lesion not found: {row['lesion']}")
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Manifest contains duplicate subject IDs')
    return rows


def cmd_one(a,r):
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_one.py')),
        '--id',r['id'],'--code',r.get('code',r['id']),'--t1',r['t1'],
        '--lesion-mode',r['lesion_mode'],
        '--out-root',a.out_root,'--tag',a.tag,'--nroi','246',
        '--transform',a.transform,'--registration-mask',a.registration_mask,
        '--margin-mm',str(a.margin_mm),'--min-voxels',str(a.min_voxels),
        '--thick-threshold-mm',str(a.thick_threshold_mm),
        '--min-brain-dice',str(a.min_brain_dice),
        '--reference-dir',a.reference_dir,'--synthstrip-cmd',a.synthstrip_cmd]
    if r['lesion_mode']=='MNI_MASK':
        cmd.extend(['--lesion',r['lesion']])
    for k in ('target','atlas','params','picked','tmp_root'):
        val=getattr(a,k)
        if val:cmd.extend(['--'+k.replace('_','-'),str(val)])
    mask=r.get('brain_mask') or (a.brain_mask if a.id else '')
    if mask:cmd+=['--brain-mask',mask]
    if a.keep_qc_images:cmd.append('--keep-qc-images')
    if a.keep_transforms:cmd.append('--keep-transforms')
    if a.allow_thick_t1:cmd.append('--allow-thick-t1')
    if a.force:cmd.append('--force')
    return cmd


def status_prior(a,r):
    base=Path(a.out_root)/r['id']/a.tag
    s=base/'qc'/'run_summary.json'
    if not s.is_file() or a.force:return 'RUN',''
    try:m=json.loads(s.read_text())
    except Exception:return 'RUN',''
    if m.get('status') not in ('SUCCESS','SUCCESS_WITH_MISSING_NODES'):
        return 'RUN',''
    expected=dict(version=VERSION,id=r['id'],t1=r['t1'],lesion=r['lesion'],
                  lesion_mode=r['lesion_mode'],tag=a.tag,nroi=246,margin_mm=a.margin_mm,min_voxels=a.min_voxels,
                  transform=a.transform,registration_mask=a.registration_mask,
                  brain_mask=r.get('brain_mask') or (a.brain_mask if a.id else ''),
                  reference_dir=a.reference_dir,allow_thick_t1=a.allow_thick_t1,
                  thick_threshold_mm=a.thick_threshold_mm,min_brain_dice=a.min_brain_dice)
    mismatch=[k for k,v in expected.items() if str(m.get(k))!=str(v)]
    if mismatch:return 'PARAM_MISMATCH',','.join(mismatch)
    outputs=[base/'network'/f'{r["id"]}_network.npy',
             base/'radiomics'/'ROI_features_246.csv',base/'radiomics'/'ROI_node_metrics.csv']
    if all(p.is_file() and p.stat().st_size>0 for p in outputs):return 'SKIP_SUCCESS','complete existing output'
    return 'RUN','incomplete existing output'


def run_subj(a,r,logs,locks):
    sid=r['id'];log=logs/f'{sid}_{a.tag}.log';lock=locks/f'{sid}_{a.tag}.lock'
    t=time.monotonic()
    with lock.open('a+') as lf:
        try:fcntl.flock(lf.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(id=sid,status='LOCKED',seconds=0,log=str(log),error='already running')
        state,detail=status_prior(a,r)
        if state!='RUN':return dict(id=sid,status=state,seconds=0,log=str(log),error=detail)
        env=os.environ.copy()
        for k in ('OMP_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            env[k]=str(a.threads_per_worker)
        with log.open('a',encoding='utf-8') as f:
            f.write('\n=== START '+time.strftime('%F %T')+' ===\n')
            f.write(' '.join(cmd_one(a,r))+'\n');f.flush()
            try:rc=subprocess.run(cmd_one(a,r),stdout=f,stderr=subprocess.STDOUT,env=env).returncode
            except Exception as e:rc=-1;f.write(repr(e)+'\n')
        summary=Path(a.out_root)/sid/a.tag/'qc'/'run_summary.json'
        state={0:'SUCCESS',2:'SUCCESS_WITH_MISSING_NODES',3:'QC_OR_FAILED'}.get(rc,'FAILED')
        detail=''
        if summary.is_file():
            try:
                m=json.loads(summary.read_text());state=m.get('status',state)
                detail=m.get('error','') or f'valid={m.get("n_valid","?")},zero={m.get("n_zero","?")},missing={m.get("n_other_missing","?")}'
            except Exception:pass
        return dict(id=sid,status=state,seconds=round(time.monotonic()-t,1),log=str(log),error=detail)


def progress(p,rows):
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['id','status','seconds','log','error'])
        wr.writeheader();wr.writerows(rows)
    tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=MANIFEST)
    p.add_argument('--out-root',default=ROOT);p.add_argument('--tag',default=TAG)
    p.add_argument('--reference-dir',default=str(Path(__file__).resolve().parent/'reference_bn246'))
    p.add_argument('--synthstrip-cmd',default='/data01/home/tanghui/lesion_seg_and_MNI/models/synthstrip/mri_synthstrip')
    p.add_argument('--transform',choices=['Affine','SyN'],default='SyN')
    p.add_argument('--registration-mask',choices=['fixed','none'],default='fixed')
    p.add_argument('--margin-mm',type=float,default=0)
    p.add_argument('--min-voxels',type=int,default=200)
    p.add_argument('--thick-threshold-mm',type=float,default=3.0)
    p.add_argument('--min-brain-dice',type=float,default=0.6)
    p.add_argument('--allow-thick-t1',action='store_true')
    p.add_argument('--brain-mask',default='',help='Only for a single --id; optional native space NIfTI brainmask')
    for k in ('target','atlas','params','picked','tmp_root'):
        p.add_argument('--'+k.replace('_','-'),default='')
    p.add_argument('--keep-qc-images',action='store_true')
    p.add_argument('--keep-transforms',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--threads-per-worker',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--id',default='')
    a=p.parse_args()
    if a.workers<1 or a.threads_per_worker<1 or a.start<0 or a.limit<0:p.error('invalid worker/start/limit')
    if a.brain_mask and not a.id:p.error('--brain-mask only allowed with --id; for cohort use optional brain_mask manifest column')
    if a.workers*a.threads_per_worker>(os.cpu_count() or 1):
        print('WARNING: workers * threads > logical CPU count',flush=True)
    if not Path(a.reference_dir,'MNI_brainmask_samegrid.nii.gz').is_file():
        raise SystemExit('Missing MNI template brainmask: first run prepare_reference.py')
    allrows=manifest_rows(a.manifest)
    if a.id:rows=[r for r in allrows if r['id']==a.id]
    else:rows=allrows[a.start:];rows=rows[:a.limit] if a.limit else rows
    if not rows:raise SystemExit('No patients selected')
    root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
    logs=root/'TMS_lesionaware_logs_v6';locks=root/'.r2sn_tms_v6_locks'
    logs.mkdir(exist_ok=True);locks.mkdir(exist_ok=True)
    pfile=root/f'{a.tag}_parallel_progress.csv'
    out=[];ctr=Counter();t=time.monotonic()
    print(f'T1prep BN246 TMS v6 n={len(rows)} workers={a.workers} threads={a.threads_per_worker}',flush=True)
    iterator=iter(rows)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        pending={}
        for _ in range(min(a.workers,len(rows))):
            row=next(iterator,None)
            if row:pending[executor.submit(run_subj,a,row,logs,locks)]=row['id']
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in finished:
                sid=pending.pop(f)
                try:r=f.result()
                except Exception as e:r=dict(id=sid,status='LAUNCH_FAILED',seconds=0,log='',error=str(e))
                out.append(r);ctr[r['status']]+=1
                print(f'[{len(out)}/{len(rows)}] {sid}: {r["status"]} done={len(out)},left={len(rows)-len(out)} '+r['error'],flush=True)
                progress(pfile,out)
                nxt=next(iterator,None)
                if nxt:pending[executor.submit(run_subj,a,nxt,logs,locks)]=nxt['id']
    print('COUNTS',dict(ctr),'\nPROGRESS_CSV',pfile,flush=True)

if __name__=='__main__':main()
