#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the SAME R2SN MNI grid, skull-stripped template and BN246 check once.
Do not replace the atlas or change the MNI reference grid.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def default_paths():
    import R2SN
    root = Path(R2SN.__file__).resolve().parent
    return root / 'data' / 'MNI152_T1_1mm.nii', root / 'data' / 'rBN_Atlas_246_1mm.nii'


def same_grid(a,b):
    return a.shape[:3]==b.shape[:3] and np.allclose(a.affine,b.affine,rtol=0,atol=0.02)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target', default='')
    p.add_argument('--atlas', default='')
    p.add_argument('--ref-dir', default=str(Path(__file__).resolve().parent/'reference_bn246'))
    p.add_argument('--synthstrip-cmd', default='/data01/home/tanghui/lesion_seg_and_MNI/models/synthstrip/mri_synthstrip')
    p.add_argument('--force', action='store_true')
    a=p.parse_args()
    dtarget,datlas=default_paths()
    target=Path(a.target or dtarget);atlas=Path(a.atlas or datlas)
    for f in (target,atlas):
        if not f.is_file():raise FileNotFoundError(str(f))
    t=nib.load(str(target));at=nib.load(str(atlas))
    if not same_grid(t,at):raise RuntimeError('R2SN MNI target and BN246 atlas grids are different')
    ad=np.asanyarray(at.dataobj)
    labels=set(np.unique(np.rint(ad).astype(np.int16)).tolist())
    if not set(range(1,247)).issubset(labels):raise RuntimeError('BN atlas missing labels 1..246')
    ref=Path(a.ref_dir);ref.mkdir(parents=True,exist_ok=True)
    m=ref/'MNI_brainmask_samegrid.nii.gz';b=ref/'MNI_brain_samegrid.nii.gz';meta=ref/'reference_meta.json'
    if m.is_file() and b.is_file() and not a.force:
        if same_grid(nib.load(str(m)),t) and same_grid(nib.load(str(b)),t):
            print('REFERENCE_ALREADY_PREPARED',ref);return
        raise RuntimeError('Existing reference grid mismatch. Inspect, then use --force to regenerate')
    cmd=shutil.which(a.synthstrip_cmd)
    if not cmd:raise RuntimeError('mri_synthstrip not found; set up FreeSurfer or pass --synthstrip-cmd executable')
    tmp_mask=ref/'_temporary_mni_mask.nii.gz';tmp_brain=ref/'_temporary_mni_brain.nii.gz'
    try:
        subprocess.run([cmd,'-i',str(target),'-m',str(tmp_mask),'-o',str(tmp_brain)],check=True)
        brainmask=nib.load(str(tmp_mask))
        if not same_grid(brainmask,t):
            brainmask=resample_from_to(brainmask,t,order=0,mode='constant',cval=0)
        mask=np.asanyarray(brainmask.dataobj)>0.5
        if mask.sum()<20000:raise RuntimeError('MNI brainmask appears implausibly small')
        td=np.asarray(t.dataobj,dtype=np.float32)
        if not np.isfinite(td).all():raise RuntimeError('MNI target includes NaN/Inf')
        nib.save(nib.Nifti1Image(mask.astype(np.uint8),t.affine),str(m))
        nib.save(nib.Nifti1Image(np.where(mask,td,0).astype('float32'),t.affine),str(b))
        meta.write_text(json.dumps({'target':str(target),'atlas':str(atlas),'mask_voxels':int(mask.sum()),
                                    'grid_shape':list(t.shape),'grid_affine':np.round(t.affine,6).tolist()},indent=2))
        print('REFERENCE_PREPARED',ref,'brain_voxels',int(mask.sum()))
        print('MASK',m,'\nBRAIN',b)
    finally:
        tmp_mask.unlink(missing_ok=True);tmp_brain.unlink(missing_ok=True)

if __name__=='__main__':main()
