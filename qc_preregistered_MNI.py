#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict QC for pre-registered MNI T1, brain mask, lesion, R2SN target and BN246."""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def load(path, name):
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f'{name} must be 3D: {path}')
    data = np.asarray(img.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return img, data


def same_grid(a, b, tol):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=tol, rtol=0)


def save_like(data, ref, path, dtype):
    hdr = ref.header.copy()
    hdr.set_data_dtype(dtype)
    out = nib.Nifti1Image(np.asarray(data, dtype=dtype), ref.affine, hdr)
    out.set_qform(ref.affine, code=1)
    out.set_sform(ref.affine, code=1)
    nib.save(out, str(path))


def plot_overlay(t1, lesion, remaining, brain, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ctr = np.rint(np.argwhere(lesion).mean(axis=0)).astype(int) if lesion.any() else np.array(t1.shape) // 2
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), layout='constrained')
    vmax = float(np.percentile(t1[brain], 99)) if brain.any() else float(np.percentile(t1, 99))
    for row, (axis, name) in enumerate(((2, 'Axial'), (1, 'Coronal'), (0, 'Sagittal'))):
        k = int(np.clip(ctr[axis], 0, t1.shape[axis] - 1))
        bg = np.rot90(np.take(t1, k, axis=axis))
        lm = np.rot90(np.take(lesion, k, axis=axis))
        rm = np.rot90(np.take(remaining > 0, k, axis=axis))
        for ax in axes[row]:
            ax.imshow(bg, cmap='gray', vmin=0, vmax=max(vmax, 1))
            ax.axis('off')
        axes[row, 0].imshow(np.ma.masked_where(~lm, lm), cmap='Reds', alpha=0.75)
        axes[row, 0].set_title(f'{name}: red = lesion')
        axes[row, 1].imshow(np.ma.masked_where(~rm, rm), cmap='Greens', alpha=0.35)
        axes[row, 1].imshow(np.ma.masked_where(~lm, lm), cmap='Reds', alpha=0.85)
        axes[row, 1].set_title(f'{name}: green = remaining BN246; red = lesion')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1', required=True)
    p.add_argument('--brainmask', required=True)
    p.add_argument('--lesion', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--atlas', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--grid-tolerance-mm', type=float, default=0.02)
    p.add_argument('--min-voxels', type=int, default=200)
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t1_img, t1 = load(a.t1, 'T1')
    bm_img, bm_data = load(a.brainmask, 'Brain mask')
    le_img, le_data = load(a.lesion, 'Lesion')
    target_img, _ = load(a.target, 'R2SN target')
    atlas_img, atlas_data = load(a.atlas, 'BN246 atlas')

    for name, img in [('T1', t1_img), ('Brain mask', bm_img), ('Lesion', le_img), ('BN246 atlas', atlas_img)]:
        ok = same_grid(img, target_img, a.grid_tolerance_mm)
        print(f'{name} matches R2SN target grid: {ok}')
        if not ok:
            raise RuntimeError(f'{name} does not match R2SN target grid. Stop; do not run R2SN.')

    brain = bm_data > 0.5
    lesion = le_data > 0
    atlas_rounded = np.rint(atlas_data).astype(np.int16)
    if not np.allclose(atlas_data, atlas_rounded, atol=1e-3):
        raise ValueError('BN246 atlas has non-integer labels')
    missing = [i for i in range(1, 247) if not np.any(atlas_rounded == i)]
    if missing:
        raise ValueError(f'BN246 atlas missing labels: {missing}')
    if not brain.any():
        raise ValueError('Brain mask is empty')
    if not lesion.any():
        raise ValueError('Lesion mask is empty; use a different mode only if this is a confirmed no-lesion subject')

    inside = lesion & brain
    inside_fraction = float(inside.sum() / max(1, lesion.sum()))
    voxel_volume = abs(float(np.linalg.det(target_img.affine[:3, :3])))
    print('\n===== INPUT QC =====')
    print('T1 shape:', t1_img.shape[:3])
    print('T1 voxel size (mm):', t1_img.header.get_zooms()[:3])
    print('Orientation:', nib.aff2axcodes(t1_img.affine))
    print('Brain voxels:', int(brain.sum()))
    print('Lesion voxels:', int(lesion.sum()))
    print('Lesion inside brain:', int(inside.sum()))
    print('Lesion inside brain fraction:', round(inside_fraction, 4))
    print('Lesion volume inside brain (mm3):', round(float(inside.sum() * voxel_volume), 2))
    print('T1 brain p1/p50/p95:', np.percentile(t1[brain], [1, 50, 95]).tolist())

    remaining = atlas_rounded.copy()
    remaining[~brain | lesion] = 0
    rows = []
    for roi in range(1, 247):
        total = atlas_rounded == roi
        in_brain = total & brain
        in_lesion = in_brain & lesion
        remain = total & brain & ~lesion
        n_total = int(total.sum())
        n_brain = int(in_brain.sum())
        n_lesion = int(in_lesion.sum())
        n_remain = int(remain.sum())
        if n_total == 0:
            status = 'ATLAS_MISSING'
        elif n_brain == 0:
            status = 'NO_BRAIN_COVERAGE'
        elif n_remain == 0 and n_lesion == n_brain:
            status = 'FULLY_LESIONED_WITHIN_BRAIN'
        elif n_remain == 0:
            status = 'NO_REMAINING_VOXELS'
        elif n_remain < a.min_voxels:
            status = 'LOW_REMAINING_VOXELS'
        else:
            status = 'OK'
        rows.append({
            'ROI': roi,
            'atlas_voxels': n_total,
            'brain_voxels': n_brain,
            'lesion_voxels_in_brain': n_lesion,
            'remaining_voxels': n_remain,
            'brain_coverage_fraction': n_brain / n_total if n_total else 0,
            'lesion_fraction_within_brain': n_lesion / n_brain if n_brain else 0,
            'status': status,
        })

    save_like(inside.astype(np.uint8), target_img, out / 'lesion_inside_brain.nii.gz', np.uint8)
    save_like(remaining, target_img, out / 'BN246_remainingROI_MNI.nii.gz', np.int16)
    with (out / 'ROI_coverage_QC.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_overlay(t1, lesion, remaining, brain, out / 'T1_lesion_BN246_overlay.png')

    print('\n===== ROI SUMMARY =====')
    for status in sorted(set(r['status'] for r in rows)):
        print(status, sum(r['status'] == status for r in rows))
    print('\nQC_COMPLETE:', out)
    if inside_fraction < 0.80:
        print('WARNING: less than 80% of lesion voxels are inside the supplied brain mask; inspect overlay before R2SN.')


if __name__ == '__main__':
    main()
