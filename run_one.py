#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TMS native T1 -> SynthStrip -> masked N4 -> optional lesion-aware ANTs -> BN246 R2SN.

IMPORTANT: This TMS v6 variant is NOT interchangeable with C3 v4/v5 results. Six-mm
2D T1 cannot become genuine isotropic 1-mm 3D T1 through interpolation.
Completely lesion-destroyed ROI => zero-coded edges; no-brain-coverage => NaN.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt
from network_utils import classify_rois, valid_feature_network, nodal_metrics, write_rows, atomic_save_npy

VERSION='TMS_BN246_native_T1prep_brainmask_v6.1'
DEFAULT_ROOT='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects'
DEFAULT_TAG='TMS_BN246_T1prep_v6'


def pkg_defaults():
    import R2SN
    root=Path(R2SN.__file__).resolve().parent
    return dict(target=root/'data'/'MNI152_T1_1mm.nii',
                atlas=root/'data'/'rBN_Atlas_246_1mm.nii',
                params=root/'network_feature'/'Params.yaml',
                picked=root/'network_feature'/'picked_features.txt')


def same_grid(a,b,tol=.02):
    return a.shape[:3]==b.shape[:3] and np.allclose(a.affine,b.affine,atol=tol,rtol=0)


def checksum(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def save_like(data,ref,p,dtype):
    hdr=ref.header.copy();hdr.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(np.asarray(data,dtype=dtype),ref.affine,hdr),str(p))


def binarize_lesion(p,target):
    src=nib.load(str(p))
    if len(src.shape)!=3:raise ValueError('MNI lesion mask must be 3D')
    source_info=dict(lesion_source_shape=list(src.shape[:3]),
                     lesion_source_voxel_mm=[float(x) for x in src.header.get_zooms()[:3]],
                     lesion_source_orientation=list(nib.aff2axcodes(src.affine)),
                     lesion_source_affine=np.asarray(src.affine,dtype=float).tolist())
    res=not same_grid(src,target)
    if res:src=resample_from_to(src,target,order=0,mode='constant',cval=0)
    ar=np.asarray(src.dataobj)
    if not np.isfinite(ar).all():raise ValueError('MNI lesion mask has nonfinite values')
    mask=ar>0
    if not mask.any():raise ValueError('Lesion empty after MNI resampling; check coordinates and template space')
    return mask,res,source_info


def margin_lesion(mask,ref,mm):
    if mm<0:raise ValueError('margin-mm must be >=0')
    if mm==0 or not np.asarray(mask,dtype=bool).any():return mask.copy()
    return distance_transform_edt(~mask,sampling=nib.affines.voxel_sizes(ref.affine)[:3])<=mm


def create_native_mask(t1,work,args):
    raw=nib.load(str(t1))
    brainmask=work/'T1_brain_mask_native.nii.gz'
    if args.brain_mask:
        supplied=nib.load(str(args.brain_mask))
        if not same_grid(supplied,raw):
            raise ValueError('User-supplied T1 brainmask is not aligned to native T1: reorient/register first')
        mask=np.asanyarray(supplied.dataobj)>0.5
        save_like(mask,raw,brainmask,np.uint8)
    else:
        cmd=shutil.which(args.synthstrip_cmd)
        if not cmd:
            raise RuntimeError('mri_synthstrip unavailable: run FreeSurfer environment, or supply --brain-mask')
        stripped=work/'T1_SynthStrip_brain.nii.gz'
        subprocess.run([cmd,'-i',str(t1),'-o',str(stripped),'-m',str(brainmask)],check=True)
        bm=nib.load(str(brainmask))
        if not same_grid(bm,raw):
            # Explicit nearest-neighbour resample, retaining actual native world-coordinate grid
            bm=resample_from_to(bm,raw,order=0,mode='constant',cval=0)
            save_like(np.asarray(bm.dataobj)>0.5,raw,brainmask,np.uint8)
        mask=np.asarray(nib.load(str(brainmask)).dataobj)>0.5
    total=int(mask.sum())
    if total<500:raise RuntimeError('SynthStrip brainmask too small; check original T1 brain extraction')
    vol=total*abs(float(np.linalg.det(raw.affine[:3,:3])))
    return brainmask, dict(native_brain_voxels=total,native_brain_volume_mm3=round(vol,2))


def register_with_brainmask(t1,brainmask,target,ref_brain,ref_mask,lesion,margin,args,work):
    import ants
    moving=ants.image_read(str(t1))
    msk=ants.image_read(str(brainmask))
    native_mask=np.asarray(msk.numpy())>0.5
    moving_corrected=ants.n4_bias_field_correction(moving,mask=msk)
    corrected_arr=np.asarray(moving_corrected.numpy(),dtype='float32')
    if not np.isfinite(corrected_arr).all():raise RuntimeError('N4 produced nonfinite values')
    # Zero the nonbrain *only after* N4, using a trusted native brain mask.
    corrected_arr[~native_mask]=0
    moving_brain=ants.from_numpy(corrected_arr,origin=moving.origin,
                                  spacing=moving.spacing,direction=moving.direction)
    fixed=ants.image_read(str(ref_brain))
    fm=ants.image_read(str(ref_mask))
    fixed_mask=np.asarray(fm.numpy())>0.5
    if args.registration_mask=='fixed':
        ref=nib.load(str(target))
        excluded=margin_lesion(lesion,ref,margin)
        # Nibabel arrays are indexed in the same RAS-native voxel grid as
        # the file read by ANTs; no manual array flip is required.
        lesion_file=work/'lesion_reg_exclude.nii.gz'
        save_like(excluded,ref,lesion_file,np.uint8)
        eants=ants.image_read(str(lesion_file))
        excluded_ants=np.asarray(eants.numpy())>0.5
        if excluded_ants.shape!=fixed_mask.shape:
            raise ValueError('Fixed lesion mask/ANTs template shape mismatch')
        fixed_mask &= ~excluded_ants
    if fixed_mask.sum()<10000:raise ValueError('Registration reference mask implausibly small')
    mask_fixed=ants.from_numpy(fixed_mask.astype('float32'),origin=fixed.origin,
                                spacing=fixed.spacing,direction=fixed.direction)
    mask_moving=ants.from_numpy(native_mask.astype('float32'),origin=moving.origin,
                                 spacing=moving.spacing,direction=moving.direction)
    reg=ants.registration(fixed=fixed,moving=moving_brain,type_of_transform=args.transform,
                          outprefix=str(work/'antsreg_'),mask=mask_fixed,
                          moving_mask=mask_moving,mask_all_stages=True)
    # Resample native brain segmentation with THE SAME fitted transforms.
    brainreg=ants.apply_transforms(fixed=fixed,moving=mask_moving,
                                   transformlist=reg['fwdtransforms'],
                                   interpolator='nearestNeighbor')
    native_brain_mni=np.asarray(brainreg.numpy())>0.5
    d=np.asarray(reg['warpedmovout'].numpy(),dtype='float32')
    if native_brain_mni.shape!=d.shape:raise ValueError('Registered brainmask/T1 shapes differ')
    if not np.isfinite(d).all():raise RuntimeError('Registered T1 has NaN or Inf')
    good=d[native_brain_mni & (d>0)]
    if good.size<10000:raise RuntimeError('Too few nonzero brain voxels after registration')
    p95=float(np.percentile(good,95))
    if p95<=0:raise RuntimeError('Registered T1 signal invalid')
    normalized=np.where(native_brain_mni,d/p95*255.0,0).astype('float32')
    t1_path=work/'registered_T1_MNI.nii.gz'
    ants.image_write(ants.from_numpy(normalized,origin=fixed.origin,
                      spacing=fixed.spacing,direction=fixed.direction),str(t1_path))
    brain_path=work/'patient_brain_mask_MNI.nii.gz'
    ants.image_write(ants.from_numpy(native_brain_mni.astype('float32'),origin=fixed.origin,
                    spacing=fixed.spacing,direction=fixed.direction),str(brain_path))
    ref_mask_img=nib.load(str(ref_mask))
    patient_brain_img=nib.load(str(brain_path))
    if not same_grid(ref_mask_img,patient_brain_img):
        raise RuntimeError('Patient brain mask warped to unexpected MNI grid')
    ref_arr=np.asarray(ref_mask_img.dataobj)>0.5
    inter=int(np.count_nonzero(native_brain_mni & ref_arr))
    dice=2*inter/max(1,int(native_brain_mni.sum()+ref_arr.sum()))
    if args.keep_transforms:
        dest=Path(args.output_base)/'qc'/'transforms'
        dest.mkdir(parents=True,exist_ok=True)
        # preserve only for this patient when explicitly requested
        for ix,path in enumerate(reg['fwdtransforms']):
            if Path(path).is_file():shutil.copy2(path,dest/f'fwd_{ix}_{Path(path).name}')
        for ix,path in enumerate(reg['invtransforms']):
            if Path(path).is_file():shutil.copy2(path,dest/f'inv_{ix}_{Path(path).name}')
    return t1_path,brain_path,float(dice)


def clip_atlas_to_brain(atlas,lesion,excluded,brain,min_voxels):
    """Only true 100%-lesion ROI is zero-coded. Missing FOV/brain is NaN."""
    clean,rows=classify_rois(atlas,lesion,excluded,246,min_voxels)
    before=clean.copy()
    clean[~brain]=0
    for r in rows:
        label=r['ROI'];prev=r['clean_voxels']
        current=int(np.count_nonzero(clean==label))
        r['clean_voxels_before_brainmask']=prev
        r['outside_patient_brain_voxels']=prev-current
        r['clean_voxels']=current
        r['brain_coverage_fraction']=(current/max(1,prev))
        if r['status'] not in ('FULLY_LESIONED_ZERO','EMPTY_AFTER_MARGIN_ZERO','ATLAS_MISSING'):
            if current==0:r['status']='NO_BRAIN_COVERAGE_MISSING'
            elif current<min_voxels:r['status']='LOW_VOXELS_ATTEMPT'
            else:r['status']='EXTRACT_PENDING'
    return clean,rows


def plot_qc(qc,t1_path,lesion_path,clean_path,brain_path,patient):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t=np.asarray(nib.load(str(t1_path)).dataobj,dtype=float)
        l=np.asarray(nib.load(str(lesion_path)).dataobj)>0
        c=np.asarray(nib.load(str(clean_path)).dataobj)>0
        b=np.asarray(nib.load(str(brain_path)).dataobj)>0
        ctr=np.rint(np.argwhere(l).mean(axis=0)).astype(int) if l.any() else np.array(t.shape)//2
        def sl(a,axis,idx):return np.rot90(np.take(a,idx,axis=axis))
        vmax=float(np.percentile(t[b],99)) if b.any() else 255
        fig,axs=plt.subplots(3,2,figsize=(10,12),layout='constrained')
        for i,(axis,name) in enumerate([(2,'Axial'),(1,'Coronal'),(0,'Sagittal')]):
            k=int(np.clip(ctr[axis],0,t.shape[axis]-1))
            bg=sl(t,axis,k); lm=sl(l,axis,k);cm=sl(c,axis,k)
            for ax in axs[i]:
                ax.imshow(bg,cmap='gray',vmin=0,vmax=max(vmax,1));ax.axis('off')
            axs[i,0].imshow(np.ma.masked_where(~lm,lm),cmap='Reds',vmin=0,vmax=1,alpha=.65)
            axs[i,0].set_title(name+': red = LESION only')
            # Green is all retained BN voxels, red is lesion, and the channels are distinct.
            axs[i,1].imshow(np.ma.masked_where(~cm,cm),cmap='Greens',vmin=0,vmax=1,alpha=.2)
            axs[i,1].imshow(np.ma.masked_where(~lm,lm),cmap='Reds',vmin=0,vmax=1,alpha=.9)
            axs[i,1].set_title(name+': green=BN ROI, red=LESION')
        fig.suptitle(f'{patient}: visually inspect lesion/T1 agreement and skull stripping')
        out=qc/'T1prep_lesion_overlay.png';fig.savefig(out,dpi=140);plt.close(fig)
        return str(out)
    except Exception as e:
        return 'QC_PNG_FAILED: '+str(e)


def signature(args,cfg):
    return dict(version=VERSION,id=args.id,code=args.code,t1=str(args.t1),
      lesion=str(args.lesion or ''),lesion_mode=args.lesion_mode,
      tag=args.tag,nroi=246,margin_mm=float(args.margin_mm),min_voxels=int(args.min_voxels),
      transform=args.transform,registration_mask=args.registration_mask,
      brain_mask=args.brain_mask,reference_dir=str(args.reference_dir),
      allow_thick_t1=bool(args.allow_thick_t1),thick_threshold_mm=float(args.thick_threshold_mm),
      min_brain_dice=float(args.min_brain_dice),
      atlas=str(cfg['atlas']),target=str(cfg['target']),
      params_sha256=checksum(cfg['params']),picked_sha256=checksum(cfg['picked']),
      fully_destroyed_policy='row_col_zero',outside_FOV_policy='NaN',
      feature_normalization='minmax_across_surviving_valid_ROIs')


def existing_ok(summary,base,meta):
    if not summary.is_file():return False
    try:old=json.loads(summary.read_text())
    except Exception:return False
    if old.get('status') not in ('SUCCESS','SUCCESS_WITH_MISSING_NODES'):return False
    mismatch=[k for k,v in meta.items() if str(old.get(k))!=str(v)]
    if mismatch:raise RuntimeError('Existing successful output has different parameters: '+str(mismatch)+'. Change --tag.')
    names=[base/'network'/f'{meta["id"]}_network.npy',
           base/'radiomics'/'ROI_features_246.csv',base/'radiomics'/'ROI_node_metrics.csv']
    if not all(p.is_file() and p.stat().st_size>0 for p in names):return False
    try:return np.load(names[0],mmap_mode='r').shape==(246,246)
    except Exception:return False


def run(args):
    from R2SN.network_feature.feature_extract import feature_extract
    defaults=pkg_defaults();cfg={k:Path(getattr(args,k) or defaults[k]) for k in defaults}
    args.lesion_mode = args.lesion_mode or ('MNI_MASK' if args.lesion else 'NO_LESION')
    if args.lesion_mode not in ('MNI_MASK','NO_LESION'):
        raise ValueError(f'Unsupported lesion_mode: {args.lesion_mode}')
    if args.lesion_mode == 'MNI_MASK' and not args.lesion:
        raise ValueError('lesion_mode=MNI_MASK but --lesion path is empty')
    if args.lesion_mode == 'NO_LESION' and args.lesion:
        raise ValueError('lesion_mode=NO_LESION must not have a lesion path')
    for p in [args.t1,*cfg.values()]:
        if not Path(p).is_file():raise FileNotFoundError(str(p))
    if args.lesion_mode == 'MNI_MASK' and not Path(args.lesion).is_file():
        raise FileNotFoundError(f'MNI lesion mask not found: {args.lesion}')
    ref_dir=Path(args.reference_dir)
    ref_brain=ref_dir/'MNI_brain_samegrid.nii.gz';ref_mask=ref_dir/'MNI_brainmask_samegrid.nii.gz'
    for p in (ref_brain,ref_mask):
        if not p.is_file():raise FileNotFoundError(f'{p}; run prepare_reference.py first')
    base=Path(args.out_root)/args.id/args.tag
    args.output_base=str(base)
    qc=base/'qc';rad=base/'radiomics';net=base/'network'
    for d in (qc,rad,net):d.mkdir(parents=True,exist_ok=True)
    summary=qc/'run_summary.json'
    meta=signature(args,cfg)
    if not args.force and existing_ok(summary,base,meta):
        print(f'[{args.id}] SKIP_EXISTING_SUCCESS');return 'SKIP_EXISTING_SUCCESS'
    meta.update(status='STARTED',time_start=time.strftime('%F %T'))
    try:
        t1=nib.load(str(args.t1)); target=nib.load(str(cfg['target']));atlas_img=nib.load(str(cfg['atlas']))
        if len(t1.shape)!=3:raise ValueError('T1 must be 3D NIfTI')
        for name,p in [('atlas',atlas_img),('reference brain',nib.load(str(ref_brain))),('reference mask',nib.load(str(ref_mask)))]:
            if not same_grid(target,p):raise ValueError(f'{name} target grid mismatch')
        vox=np.asarray(t1.header.get_zooms()[:3],dtype=float)
        if not np.isfinite(vox).all() or np.min(vox)<=0:raise ValueError('T1 header voxel size invalid')
        meta['t1_shape']=list(t1.shape);meta['t1_voxel_mm']=vox.tolist()
        meta['t1_orientation']=list(nib.aff2axcodes(t1.affine))
        meta['thick_t1']=bool(float(vox.max())>args.thick_threshold_mm)
        meta['radiomics_thickness_caution']='THICK_LAYER_INTERPOLATED_TEXTURE_NOT_EQUIVALENT_TO_3D_T1' if meta['thick_t1'] else ''
        if meta['thick_t1'] and not args.allow_thick_t1:
            meta['status']='QC_THICK_T1_HOLD'
            print(f'[{args.id}] QC_THICK_T1_HOLD: voxel {vox.tolist()} mm. '
                  'To test only, add --allow-thick-t1.',flush=True)
            return meta['status']
        atlas_raw=np.asarray(atlas_img.dataobj)
        if not np.isfinite(atlas_raw).all() or not np.allclose(atlas_raw,np.rint(atlas_raw),atol=1e-3):
            raise ValueError('BN atlas must be integer-labeled')
        atlas=np.rint(atlas_raw).astype(np.int16)
        if any(not (atlas==i).any() for i in range(1,247)):
            raise ValueError('Missing BN246 atlas labels')
        if args.lesion_mode == 'MNI_MASK':
            lesion,was_resampled,lesion_source_info=binarize_lesion(args.lesion,target)
            meta.update(lesion_source_info)
        else:
            # Healthy/no-lesion subject: an all-zero mask on the exact R2SN target grid.
            lesion=np.zeros(target.shape[:3],dtype=bool)
            was_resampled=False
            meta['lesion_source_shape']=[]
            meta['lesion_source_voxel_mm']=[]
            meta['lesion_source_orientation']=[]
            meta['lesion_source_affine']=[]
        expanded=margin_lesion(lesion,target,args.margin_mm)
        meta['lesion_mode']=args.lesion_mode
        meta['lesion_resampled']=bool(was_resampled)
        meta['lesion_space_assumption']='MNI lesion and R2SN target assumed to share compatible MNI world coordinates; resampling does not correct template-space mismatch'
        meta['lesion_volume_mm3']=float(lesion.sum()*abs(np.linalg.det(target.affine[:3,:3])))
        meta['lesion_voxels_target_grid']=int(lesion.sum())
        meta['atlas_lesion_overlap_voxels']=int(np.count_nonzero((atlas>0)&(atlas<=246)&lesion))
        with tempfile.TemporaryDirectory(prefix='R2SN_T1prep_'+args.id+'_',dir=args.tmp_root) as tempdir:
            work=Path(tempdir)
            mask,mask_info=create_native_mask(args.t1,work,args)
            meta.update(mask_info)
            registered,brain_path,dice=register_with_brainmask(
                 args.t1,mask,cfg['target'],ref_brain,ref_mask,lesion,args.margin_mm,args,work)
            meta['registration_brain_dice']=float(dice)
            if dice<args.min_brain_dice:
                meta['status']='QC_REGISTRATION_HOLD'
                print(f'[{args.id}] QC_REGISTRATION_HOLD: Dice={dice:.3f}',flush=True)
                if args.keep_qc_images:
                    shutil.copyfile(registered,qc/'registered_T1_MNI.nii.gz')
                    shutil.copyfile(brain_path,qc/'patient_brain_mask_MNI.nii.gz')
                    save_like(lesion,target,qc/'lesion_resampled_MNI.nii.gz',np.uint8)
                return meta['status']
            brain_img=nib.load(str(brain_path))
            if not same_grid(brain_img,target):raise ValueError('Warped patient brain not on target grid')
            patient_brain=np.asarray(brain_img.dataobj)>0.5
            clean,rows=clip_atlas_to_brain(atlas,lesion,expanded,patient_brain,args.min_voxels)
            meta['n_lesioned_rois']=sum(r['lesion_voxels']>0 for r in rows)
            meta['n_fully_lesioned_rois']=sum(r['status']=='FULLY_LESIONED_ZERO' for r in rows)
            meta['n_zero_brain_coverage']=sum(r['status']=='NO_BRAIN_COVERAGE_MISSING' for r in rows)
            meta['n_low_voxel_rois']=sum(r['status']=='LOW_VOXELS_ATTEMPT' for r in rows)
            write_rows(qc/'ROI_lesion_overlap.csv',rows)
            clean_path=work/'BN246_remaining_ROI_MNI.nii.gz'
            save_like(clean,target,clean_path,np.int16)
            if args.keep_qc_images:
                shutil.copyfile(registered,qc/'registered_T1_MNI.nii.gz')
                shutil.copyfile(brain_path,qc/'patient_brain_mask_MNI.nii.gz')
                shutil.copyfile(mask,qc/'T1_brain_mask_native.nii.gz')
                save_like(lesion,target,qc/'lesion_resampled_MNI.nii.gz',np.uint8)
                save_like(clean,target,qc/'BN_lesionExcluded.nii.gz',np.int16)
                meta['qc_png']=plot_qc(qc,registered,qc/'lesion_resampled_MNI.nii.gz',
                                        qc/'BN_lesionExcluded.nii.gz',brain_path,args.id)
            candidates=[r['ROI'] for r in rows if r['clean_voxels']>0]
            csvpath=work/'radiomics_input.csv'
            with csvpath.open('w',newline='',encoding='utf-8') as f:
                wr=csv.writer(f);wr.writerow(['Image','Mask','Label'])
                for i in candidates:wr.writerow([str(registered),str(clean_path),i])
            rtemp=work/'features';rtemp.mkdir()
            print(f'[{args.id}] Extracting radiomics from {len(candidates)}/246 remaining ROIs',flush=True)
            feat=pd.DataFrame()
            feature_csv=None
            if candidates:
                feature_name=feature_extract(str(csvpath),str(cfg['params']),str(rtemp))
                feature_csv=rtemp/feature_name
                if feature_csv.is_file():feat=pd.read_csv(feature_csv)
            picked=ast.literal_eval(cfg['picked'].read_text(encoding='utf-8'))
            W,valid,full_feat,states,stats=valid_feature_network(feat,picked,rows,246)
            for row,s in zip(rows,states):row['status']=s
            write_rows(qc/'ROI_lesion_overlap.csv',rows)
            meta.update(stats)
            meta['n_rois_lesion_fraction_over_5pct']=sum(r['lesion_fraction']>.05 for r in rows)
            meta['feature_rows_requested']=len(candidates);meta['feature_rows_extracted']=len(feat)
            pfeat=pd.DataFrame(full_feat,columns=picked)
            pfeat.insert(0,'status',states);pfeat.insert(0,'ROI',list(range(1,247)))
            pfeat.to_csv(rad/'ROI_features_246.csv',index=False)
            if feature_csv and feature_csv.is_file():
                # This features CSV contains temporary full paths, but numeric features remain valid.
                shutil.copyfile(feature_csv,rad/'pyradiomics_input_radiomics_features.csv')
            write_rows(rad/'ROI_node_metrics.csv',nodal_metrics(W,valid,rows,states))
            atomic_save_npy(net/f'{args.id}_network.npy',W)
            atomic_save_npy(net/f'{args.id}_valid_roi_mask.npy',valid)
            atomic_save_npy(net/f'{args.id}_network_zeroFilled_DISPLAY_ONLY.npy',np.nan_to_num(W,nan=0.0))
            meta['network_path']=str(net/f'{args.id}_network.npy')
            meta['status']='SUCCESS_WITH_MISSING_NODES' if stats['n_other_missing'] else 'SUCCESS'
            return meta['status']
    except Exception as e:
        meta['status']='FAILED'
        meta['error']=f'{type(e).__name__}: {e}'
        return meta['status']
    finally:
        meta['time_end']=time.strftime('%F %T')
        tmp=summary.with_name(summary.name+'.tmp')
        tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        tmp.replace(summary)
        print(f'[{args.id}] {meta["status"]} valid={meta.get("n_valid",0)} '
              f'zero={meta.get("n_zero",0)} missing={meta.get("n_other_missing",0)} '
              f'{meta.get("error","")}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True);p.add_argument('--code',default='')
    p.add_argument('--t1',required=True);p.add_argument('--lesion',default='')
    p.add_argument('--lesion-mode',choices=['MNI_MASK','NO_LESION'],default='')
    p.add_argument('--out-root',default=DEFAULT_ROOT)
    p.add_argument('--tag',default=DEFAULT_TAG)
    p.add_argument('--nroi',type=int,choices=[246],default=246)
    for x in ['target','atlas','params','picked','tmp_root']:
        p.add_argument('--'+x.replace('_','-'),default='')
    p.add_argument('--reference-dir',default=str(Path(__file__).resolve().parent/'reference_bn246'))
    p.add_argument('--brain-mask',default='',help='Optional native T1-space mask: skips running SynthStrip')
    p.add_argument('--synthstrip-cmd',default='/data01/home/tanghui/lesion_seg_and_MNI/models/synthstrip/mri_synthstrip')
    p.add_argument('--transform',choices=['Affine','SyN'],default='SyN')
    p.add_argument('--registration-mask',choices=['fixed','none'],default='fixed')
    p.add_argument('--margin-mm',type=float,default=0)
    p.add_argument('--min-voxels',type=int,default=200)
    p.add_argument('--thick-threshold-mm',type=float,default=3.0)
    p.add_argument('--allow-thick-t1',action='store_true',
                   help='Research pilot override: 6mm T1 is not equivalent to true 1mm anatomical imaging')
    p.add_argument('--min-brain-dice',type=float,default=0.6)
    p.add_argument('--keep-qc-images',action='store_true')
    p.add_argument('--keep-transforms',action='store_true')
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    if a.min_voxels<1 or not 0<a.min_brain_dice<1:p.error('min-voxels>=1 and 0<min-brain-dice<1 required')
    state=run(a)
    raise SystemExit(0 if state in ('SUCCESS','SKIP_EXISTING_SUCCESS') else 2 if state=='SUCCESS_WITH_MISSING_NODES' else 3)

if __name__=='__main__':main()
