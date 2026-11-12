#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2SN BN246 runner for T1/brainmask/lesion already aligned to the R2SN MNI grid.

This runner intentionally performs NO SynthStrip and NO ANTs registration.
It validates that T1, brain mask, lesion, R2SN target and BN246 atlas share
one voxel grid. It optionally applies masked N4 bias correction in-place in
MNI space, then uses the original R2SN feature extraction and network code.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import ants
import nibabel as nib
import numpy as np
import pandas as pd

from run_one import (
    pkg_defaults, same_grid, save_like, margin_lesion,
    clip_atlas_to_brain, plot_qc,
)
from network_utils import (
    valid_feature_network, nodal_metrics, write_rows, atomic_save_npy,
)

VERSION = 'TMS_BN246_preRegistered_MNI_v7.0'
DEFAULT_ROOT = '/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects'
DEFAULT_TAG = 'TMS_BN246_preRegistered_v7'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_3d(path, name):
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f'{name} must be a 3D NIfTI: {path}')
    data = np.asarray(img.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f'{name} contains NaN/Inf: {path}')
    return img, data


def assert_same_grid(name, img, target, tol):
    if img.shape[:3] != target.shape[:3]:
        raise ValueError(
            f'{name} shape {img.shape[:3]} != R2SN target {target.shape[:3]}. '
            'Do not run Radiomics until the spatial grid is reconciled.'
        )
    if not np.allclose(img.affine, target.affine, atol=tol, rtol=0):
        raise ValueError(
            f'{name} affine differs from R2SN target (tolerance={tol} mm). '
            'A shared MNI label is not enough: verify template/world coordinates. '
            'Do not fix a template-space mismatch by resampling alone.'
        )


def preprocess_registered_t1(t1_path, brainmask_path, out_path, skip_n4=False):
    """Masked N4 and p95 intensity normalization; no spatial transform."""
    moving = ants.image_read(str(t1_path))
    mask_img = ants.image_read(str(brainmask_path))
    mask = np.asarray(mask_img.numpy()) > 0.5
    if mask.sum() < 5000:
        raise ValueError(f'Brain mask implausibly small: {int(mask.sum())} voxels')

    if skip_n4:
        corrected = moving
    else:
        corrected = ants.n4_bias_field_correction(moving, mask=mask_img)

    data = np.asarray(corrected.numpy(), dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('N4/T1 data contains NaN or Inf')
    good = data[mask & (data > 0)]
    if good.size < 5000:
        raise ValueError('Too few positive T1 voxels inside supplied brain mask')
    p95 = float(np.percentile(good, 95))
    if not np.isfinite(p95) or p95 <= 0:
        raise ValueError(f'Invalid T1 p95 intensity: {p95}')

    normalized = np.where(mask, data / p95 * 255.0, 0).astype(np.float32)
    out_img = ants.from_numpy(
        normalized,
        origin=moving.origin,
        spacing=moving.spacing,
        direction=moving.direction,
    )
    ants.image_write(out_img, str(out_path))
    return {
        'n4_applied': not skip_n4,
        'intensity_normalization': 'brain-mask p95 -> 255',
        't1_brain_p95_before_scaling': p95,
        't1_positive_brain_voxels': int(good.size),
    }


def make_signature(args, cfg):
    return {
        'version': VERSION,
        'id': args.id,
        'code': args.code,
        't1': str(args.t1),
        'brain_mask': str(args.brain_mask),
        'lesion': str(args.lesion or ''),
        'lesion_mode': args.lesion_mode,
        'tag': args.tag,
        'nroi': 246,
        'margin_mm': float(args.margin_mm),
        'min_voxels': int(args.min_voxels),
        'skip_n4': bool(args.skip_n4),
        'grid_tolerance_mm': float(args.grid_tolerance_mm),
        'atlas': str(cfg['atlas']),
        'target': str(cfg['target']),
        'params_sha256': sha256_file(cfg['params']),
        'picked_sha256': sha256_file(cfg['picked']),
        't1_sha256': sha256_file(args.t1),
        'brain_mask_sha256': sha256_file(args.brain_mask),
        'lesion_sha256': sha256_file(args.lesion) if args.lesion else '',
        'spatial_processing': 'PRE_REGISTERED_MNI_NO_SYN_NO_RESAMPLING',
        'fully_destroyed_policy': 'row_col_zero',
        'nonempty_unextractable_policy': 'NaN',
        'feature_normalization': 'minmax_across_surviving_valid_ROIs',
    }


def existing_ok(summary, base, meta):
    if not summary.is_file():
        return False
    try:
        old = json.loads(summary.read_text(encoding='utf-8'))
    except Exception:
        return False
    if old.get('status') not in ('SUCCESS', 'SUCCESS_WITH_MISSING_NODES'):
        return False
    mismatch = [k for k, v in meta.items() if str(old.get(k)) != str(v)]
    if mismatch:
        raise RuntimeError(
            'Existing successful output has different parameters: '
            + ', '.join(mismatch) + '. Use a new --tag or --force.'
        )
    outputs = [
        base / 'network' / f'{meta["id"]}_network.npy',
        base / 'network' / f'{meta["id"]}_valid_roi_mask.npy',
        base / 'radiomics' / 'ROI_features_246.csv',
        base / 'radiomics' / 'ROI_node_metrics.csv',
    ]
    if not all(p.is_file() and p.stat().st_size > 0 for p in outputs):
        return False
    try:
        return np.load(outputs[0], mmap_mode='r').shape == (246, 246)
    except Exception:
        return False


def run(args):
    from R2SN.network_feature.feature_extract import feature_extract

    defaults = pkg_defaults()
    cfg = {k: Path(getattr(args, k) or defaults[k]) for k in defaults}
    args.lesion_mode = args.lesion_mode or ('MNI_MASK' if args.lesion else 'NO_LESION')
    if args.lesion_mode not in ('MNI_MASK', 'NO_LESION'):
        raise ValueError(f'Unsupported lesion_mode: {args.lesion_mode}')
    if args.lesion_mode == 'MNI_MASK' and not args.lesion:
        raise ValueError('MNI_MASK requires --lesion')
    if args.lesion_mode == 'NO_LESION' and args.lesion:
        raise ValueError('NO_LESION must not have --lesion')

    required = [args.t1, args.brain_mask, *cfg.values()]
    if args.lesion_mode == 'MNI_MASK':
        required.append(args.lesion)
    for p in required:
        if not Path(p).is_file():
            raise FileNotFoundError(str(p))

    if args.tmp_root:
        Path(args.tmp_root).mkdir(parents=True, exist_ok=True)

    base = Path(args.out_root) / args.id / args.tag
    qc, rad, net = base / 'qc', base / 'radiomics', base / 'network'
    for d in (qc, rad, net):
        d.mkdir(parents=True, exist_ok=True)
    summary = qc / 'run_summary.json'

    meta = make_signature(args, cfg)
    if not args.force and existing_ok(summary, base, meta):
        print(f'[{args.id}] SKIP_EXISTING_SUCCESS', flush=True)
        return 'SKIP_EXISTING_SUCCESS'

    meta.update(status='STARTED', time_start=time.strftime('%F %T'))
    try:
        t1_img, t1_data = load_3d(args.t1, 'Pre-registered T1')
        brain_img, brain_data = load_3d(args.brain_mask, 'Brain mask')
        target_img, _ = load_3d(cfg['target'], 'R2SN target')
        atlas_img, atlas_raw = load_3d(cfg['atlas'], 'BN246 atlas')

        assert_same_grid('T1', t1_img, target_img, args.grid_tolerance_mm)
        assert_same_grid('Brain mask', brain_img, target_img, args.grid_tolerance_mm)
        assert_same_grid('BN246 atlas', atlas_img, target_img, args.grid_tolerance_mm)

        if args.lesion_mode == 'MNI_MASK':
            lesion_img, lesion_data = load_3d(args.lesion, 'Lesion mask')
            assert_same_grid('Lesion mask', lesion_img, target_img, args.grid_tolerance_mm)
            lesion = lesion_data > 0
            if not lesion.any():
                raise ValueError('Lesion mask is empty')
        else:
            lesion = np.zeros(target_img.shape[:3], dtype=bool)

        brain = brain_data > 0.5
        if int(brain.sum()) < 5000:
            raise ValueError(f'Brain mask too small: {int(brain.sum())} voxels')
        if not np.any(t1_data[brain] > 0):
            raise ValueError('T1 has no positive intensity inside brain mask')

        vox = np.asarray(t1_img.header.get_zooms()[:3], dtype=float)
        if not np.isfinite(vox).all() or np.min(vox) <= 0:
            raise ValueError('T1 voxel size invalid')
        meta.update({
            't1_shape': list(t1_img.shape[:3]),
            't1_voxel_mm': vox.tolist(),
            't1_orientation': list(nib.aff2axcodes(t1_img.affine)),
            'brain_voxels': int(brain.sum()),
            'lesion_voxels': int(lesion.sum()),
            'lesion_volume_mm3': float(lesion.sum() * abs(np.linalg.det(target_img.affine[:3, :3]))),
            'lesion_inside_brain_voxels': int(np.count_nonzero(lesion & brain)),
            'lesion_inside_brain_fraction': float(np.count_nonzero(lesion & brain) / max(1, lesion.sum())),
            'lesion_atlas_overlap_voxels': int(np.count_nonzero(lesion & (atlas_raw > 0) & (atlas_raw <= 246))),
            'target_shape': list(target_img.shape[:3]),
            'target_orientation': list(nib.aff2axcodes(target_img.affine)),
        })
        if max(vox) > args.thick_threshold_mm and not args.allow_thick_t1:
            meta['status'] = 'QC_THICK_T1_HOLD'
            print(f'[{args.id}] QC_THICK_T1_HOLD: voxel size={vox.tolist()} mm; inspect and explicitly override only for pilot.', flush=True)
            return meta['status']
        if meta['lesion_inside_brain_fraction'] < args.min_lesion_inside_brain_fraction and lesion.any():
            meta['lesion_brain_overlap_warning'] = True
            print(
                f'[{args.id}] WARNING: only {meta["lesion_inside_brain_fraction"]:.3f} '
                'of lesion voxels overlap the supplied brain mask; inspect overlay.',
                flush=True,
            )
        else:
            meta['lesion_brain_overlap_warning'] = False

        if not np.isfinite(atlas_raw).all() or not np.allclose(atlas_raw, np.rint(atlas_raw), atol=1e-3):
            raise ValueError('BN246 atlas must contain integer labels')
        atlas = np.rint(atlas_raw).astype(np.int16)
        missing = [i for i in range(1, 247) if not np.any(atlas == i)]
        if missing:
            raise ValueError(f'BN246 atlas missing labels: {missing}')

        expanded = margin_lesion(lesion, target_img, args.margin_mm)

        with tempfile.TemporaryDirectory(
            prefix=f'R2SN_preReg_{args.id}_', dir=args.tmp_root or None
        ) as tempdir:
            work = Path(tempdir)
            normalized_t1 = work / 'registered_T1_MNI.nii.gz'
            prep_info = preprocess_registered_t1(
                args.t1, args.brain_mask, normalized_t1, skip_n4=args.skip_n4
            )
            meta.update(prep_info)

            clean, rows = clip_atlas_to_brain(
                atlas, lesion, expanded, brain, args.min_voxels
            )
            meta['n_lesioned_rois'] = sum(r['lesion_voxels'] > 0 for r in rows)
            meta['n_fully_lesioned_rois'] = sum(r['status'] == 'FULLY_LESIONED_ZERO' for r in rows)
            meta['n_zero_brain_coverage'] = sum(r['status'] == 'NO_BRAIN_COVERAGE_MISSING' for r in rows)
            meta['n_low_voxel_rois'] = sum(r['status'] == 'LOW_VOXELS_ATTEMPT' for r in rows)
            meta['n_rois_lesion_fraction_over_5pct'] = sum(r['lesion_fraction'] > 0.05 for r in rows)
            meta['n_rois_lesion_fraction_over_20pct'] = sum(r['lesion_fraction'] > 0.20 for r in rows)

            clean_path = work / 'BN246_remaining_ROI_MNI.nii.gz'
            save_like(clean, target_img, clean_path, np.int16)

            if args.keep_qc_images:
                shutil.copy2(normalized_t1, qc / 'registered_T1_MNI.nii.gz')
                shutil.copy2(args.brain_mask, qc / 'patient_brain_mask_MNI.nii.gz')
                if args.lesion_mode == 'MNI_MASK':
                    shutil.copy2(args.lesion, qc / 'lesion_resampled_MNI.nii.gz')
                else:
                    save_like(lesion, target_img, qc / 'lesion_resampled_MNI.nii.gz', np.uint8)
                shutil.copy2(clean_path, qc / 'BN_lesionExcluded.nii.gz')
                meta['qc_png'] = plot_qc(
                    qc, normalized_t1, qc / 'lesion_resampled_MNI.nii.gz',
                    qc / 'BN_lesionExcluded.nii.gz',
                    qc / 'patient_brain_mask_MNI.nii.gz', args.id,
                )

            candidates = [r['ROI'] for r in rows if r['clean_voxels'] > 0]
            csv_path = work / 'radiomics_input.csv'
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                wr = csv.writer(f)
                wr.writerow(['Image', 'Mask', 'Label'])
                for roi in candidates:
                    wr.writerow([str(normalized_t1), str(clean_path), roi])

            feature_dir = work / 'features'
            feature_dir.mkdir()
            feature_df = pd.DataFrame()
            feature_csv = None
            print(
                f'[{args.id}] PRE-REGISTERED MNI: no SynthStrip, no registration; '
                f'extracting {len(candidates)}/246 remaining ROIs',
                flush=True,
            )
            if candidates:
                feature_name = feature_extract(str(csv_path), str(cfg['params']), str(feature_dir))
                feature_csv = feature_dir / feature_name
                if feature_csv.is_file():
                    feature_df = pd.read_csv(feature_csv)
                else:
                    raise FileNotFoundError(f'R2SN feature_extract did not create: {feature_csv}')

            picked = ast.literal_eval(cfg['picked'].read_text(encoding='utf-8'))
            W, valid, full_features, states, stats = valid_feature_network(
                feature_df, picked, rows, 246
            )
            for row, state in zip(rows, states):
                row['status'] = state
            write_rows(qc / 'ROI_lesion_overlap.csv', rows)
            meta.update(stats)
            meta['feature_rows_requested'] = int(len(candidates))
            meta['feature_rows_extracted'] = int(len(feature_df))

            feature_out = pd.DataFrame(full_features, columns=picked)
            feature_out.insert(0, 'status', states)
            feature_out.insert(0, 'ROI', list(range(1, 247)))
            feature_out.to_csv(rad / 'ROI_features_246.csv', index=False)
            if feature_csv and feature_csv.is_file():
                shutil.copy2(feature_csv, rad / 'pyradiomics_input_radiomics_features.csv')
            write_rows(rad / 'ROI_node_metrics.csv', nodal_metrics(W, valid, rows, states))
            atomic_save_npy(net / f'{args.id}_network.npy', W)
            atomic_save_npy(net / f'{args.id}_valid_roi_mask.npy', valid)
            atomic_save_npy(
                net / f'{args.id}_network_zeroFilled_DISPLAY_ONLY.npy',
                np.nan_to_num(W, nan=0.0),
            )
            meta['network_path'] = str(net / f'{args.id}_network.npy')
            meta['radiomics_path'] = str(rad / 'ROI_features_246.csv')
            meta['status'] = 'SUCCESS_WITH_MISSING_NODES' if stats['n_other_missing'] else 'SUCCESS'
            return meta['status']

    except Exception as e:
        meta['status'] = 'FAILED'
        meta['error'] = f'{type(e).__name__}: {e}'
        return meta['status']
    finally:
        meta['time_end'] = time.strftime('%F %T')
        tmp = summary.with_name(summary.name + '.tmp')
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(summary)
        print(
            f'[{args.id}] {meta["status"]} '
            f'valid={meta.get("n_valid", 0)} zero={meta.get("n_zero", 0)} '
            f'missing={meta.get("n_other_missing", 0)} {meta.get("error", "")}',
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    p.add_argument('--code', default='')
    p.add_argument('--t1', required=True, help='T1_EasyReg_MNI_brain.nii.gz')
    p.add_argument('--brain-mask', required=True, help='T1_EasyReg_MNI_brainmask.nii.gz')
    p.add_argument('--lesion', default='', help='lesion_MNI_on_T1grid.nii.gz for MNI_MASK')
    p.add_argument('--lesion-mode', choices=['MNI_MASK', 'NO_LESION'], default='')
    p.add_argument('--out-root', default=DEFAULT_ROOT)
    p.add_argument('--tag', default=DEFAULT_TAG)
    p.add_argument('--nroi', type=int, choices=[246], default=246)
    for name in ('target', 'atlas', 'params', 'picked', 'tmp_root'):
        p.add_argument('--' + name.replace('_', '-'), default='')
    p.add_argument('--margin-mm', type=float, default=0.0)
    p.add_argument('--min-voxels', type=int, default=200)
    p.add_argument('--grid-tolerance-mm', type=float, default=0.02)
    p.add_argument('--min-lesion-inside-brain-fraction', type=float, default=0.80)
    p.add_argument('--thick-threshold-mm', type=float, default=3.0)
    p.add_argument('--allow-thick-t1', action='store_true')
    p.add_argument('--skip-n4', action='store_true', help='Skip masked N4; default is to apply N4')
    p.add_argument('--keep-qc-images', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if args.margin_mm < 0 or args.min_voxels < 1:
        p.error('--margin-mm must be >=0 and --min-voxels >=1')
    if args.grid_tolerance_mm < 0:
        p.error('--grid-tolerance-mm must be >=0')
    if not 0 < args.min_lesion_inside_brain_fraction <= 1:
        p.error('--min-lesion-inside-brain-fraction must be in (0,1]')
    state = run(args)
    raise SystemExit(0 if state in ('SUCCESS', 'SKIP_EXISTING_SUCCESS') else 2 if state == 'SUCCESS_WITH_MISSING_NODES' else 3)


if __name__ == '__main__':
    main()
