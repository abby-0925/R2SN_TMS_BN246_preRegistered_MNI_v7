#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent TMS v6 lesion+native brainmask ROI verification. Read-only audit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import nibabel as nib


def same_grid(a,b):
    return a.shape==b.shape and np.allclose(a.affine,b.affine,atol=.02,rtol=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--id',required=True)
    p.add_argument('--root',default='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects')
    p.add_argument('--tag',default='TMS_BN246_T1prep_v6')
    a=p.parse_args()
    base=Path(a.root)/a.id/a.tag;qc=base/'qc'
    summary=json.loads((qc/'run_summary.json').read_text())
    if not all((qc/x).is_file() for x in ['registered_T1_MNI.nii.gz',
       'patient_brain_mask_MNI.nii.gz','lesion_resampled_MNI.nii.gz','BN_lesionExcluded.nii.gz']):
        raise SystemExit('Missing QC NIfTIs: re-run with --keep-qc-images and a NEW tag or --force')
    atlas=nib.load(summary['atlas']); images={}
    for name,file in [('brain','patient_brain_mask_MNI.nii.gz'),('lesion','lesion_resampled_MNI.nii.gz'),
                      ('clean','BN_lesionExcluded.nii.gz'),('T1','registered_T1_MNI.nii.gz')]:
        images[name]=nib.load(str(qc/file))
        if not same_grid(atlas,images[name]):raise ValueError(f'MNI grid mismatch: {name}')
    aarr=np.rint(np.asarray(atlas.dataobj)).astype(np.int16)
    clean=np.rint(np.asarray(images['clean'].dataobj)).astype(np.int16)
    brain=np.asarray(images['brain'].dataobj)>0.5
    lesion=np.asarray(images['lesion'].dataobj)>0.5
    mm=float(summary['margin_mm']); excl=lesion
    if mm and lesion.any():
        from scipy.ndimage import distance_transform_edt
        excl=distance_transform_edt(~lesion,sampling=nib.affines.voxel_sizes(atlas.affine)[:3])<=mm
    expected=np.where((aarr>=1)&(aarr<=246)&brain&~excl,aarr,0).astype(np.int16)
    mismatch=int(np.count_nonzero(expected!=clean))
    leak=int(np.count_nonzero(lesion&(clean!=0)))
    outside_brain=int(np.count_nonzero(~brain&(clean!=0)))
    changed_good=int(np.count_nonzero((aarr>=1)&(aarr<=246)&brain&~excl&(clean!=aarr)))
    rows=list(csv.DictReader((qc/'ROI_lesion_overlap.csv').open(encoding='utf-8-sig')))
    bad_roi=[]
    for row in rows:
        i=int(row['ROI'])
        if int(np.count_nonzero(clean==i))!=int(row['clean_voxels']):bad_roi.append(i)
    net=np.load(base/'network'/f'{a.id}_network.npy')
    sym=net.shape==(246,246) and np.allclose(net,net.T,equal_nan=True)
    z=[int(row['ROI']) for row in rows if row['status'] in ('FULLY_LESIONED_ZERO','EMPTY_AFTER_MARGIN_ZERO')]
    wrong_zero=[i for i in z if not (np.all(net[i-1]==0) and np.all(net[:,i-1]==0))]
    print('id=',a.id,'status=',summary.get('status'))
    print('brain_dice=',summary.get('registration_brain_dice'),'T1_voxel_mm=',summary.get('t1_voxel_mm'))
    print('atlas_mismatch_voxels=',mismatch)
    print('lesion_leak_voxels=',leak)
    print('nonbrain_leak_voxels=',outside_brain)
    print('wrongly_removed_intact_voxels=',changed_good)
    print('ROI_count_mismatch=',bad_roi[:20],'total=',len(bad_roi))
    print('network=',net.shape,'symmetric=',sym,'fully_destroyed_nodes=',z,'wrong_zero=',wrong_zero)
    passed=not any([mismatch,leak,outside_brain,changed_good,bad_roi,wrong_zero]) and sym
    print('T1PREP_V5_MASK_AUDIT:', 'PASS' if passed else 'FAIL')
    print('NOTE: This cannot prove anatomical registration accuracy; inspect qc/T1prep_lesion_overlay.png')
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
