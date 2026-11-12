#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a TMS ses01 manifest: MNI lesion present -> lesion-aware; absent -> no-lesion.

T1 is never copied or modified. Native T1-space lesion masks are not used by
this pipeline; the already available MNI lesion masks are used for stroke cases.
A missing lesion file is treated as NO_LESION for processing, but the subject's
clinical group must be verified before final analysis.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_T1_ROOT = '/data02/linzhenzong/TMS-MRI/Data_flip'
DEFAULT_LESION_ROOT = '/data02/linzhenzong/TMS-MRI/Lesion/lesion_mask_MNI1mm_flip'
DEFAULT_OUT_ROOT = '/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2'


def subject_id_from_t1(path):
    # Expected BIDS-like folder/file: sub-TMS001ses01/anat/sub-TMS001ses01_T1w.nii.gz
    candidates = [path.name, path.parent.name, path.parent.parent.name]
    for value in candidates:
        m = re.search(r'(sub-TMS\d+ses\d+)', value, re.I)
        if m:
            return m.group(1)
    return None


def subject_id_from_lesion(path):
    # Expected: sub-TMS001ses01_space-MNI152NLin6Asym_res-01_label-lesion_mask.nii.gz
    m = re.match(r'^(sub-TMS\d+ses\d+)_space-.*_label-lesion_mask\.nii(?:\.gz)?$', path.name, re.I)
    if m:
        return m.group(1)
    # Fallback: subject identifier before the first _space-
    if '_space-' in path.name:
        candidate = path.name.split('_space-', 1)[0]
        if re.fullmatch(r'sub-TMS\d+ses\d+', candidate, re.I):
            return candidate
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1-root', default=DEFAULT_T1_ROOT)
    p.add_argument('--lesion-root', default=DEFAULT_LESION_ROOT)
    p.add_argument('--out-root', default=DEFAULT_OUT_ROOT)
    p.add_argument('--pattern', default='sub-TMS*ses01/anat/*_T1w.nii.gz')
    p.add_argument('--clinical-table', default='', help='Optional CSV with subject_id and group columns')
    p.add_argument('--clinical-id-col', default='subject_id')
    p.add_argument('--clinical-group-col', default='group')
    a = p.parse_args()

    t1_root, lesion_root, out_root = map(Path, (a.t1_root, a.lesion_root, a.out_root))
    if not t1_root.is_dir():
        raise SystemExit(f'T1 directory not found: {t1_root}')
    if not lesion_root.is_dir():
        raise SystemExit(f'MNI lesion directory not found: {lesion_root}')

    t1_files = sorted(t1_root.glob(a.pattern))
    if not t1_files:
        raise SystemExit(f'No T1 files found: {t1_root}/{a.pattern}')

    lesion_index = {}
    duplicate_lesions = set()
    for f in sorted(lesion_root.rglob('*label-lesion_mask.nii*')):
        if not f.is_file():
            continue
        sid = subject_id_from_lesion(f)
        if not sid:
            continue
        if sid in lesion_index:
            duplicate_lesions.add(sid)
        else:
            lesion_index[sid] = f

    clinical = {}
    if a.clinical_table:
        cp = Path(a.clinical_table)
        with cp.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                sid = row.get(a.clinical_id_col, '').strip()
                grp = row.get(a.clinical_group_col, '').strip()
                if sid:
                    if sid in clinical:
                        raise SystemExit(f'Duplicate subject in clinical table: {sid}')
                    clinical[sid] = grp

    # Preflight duplicate IDs before writing any manifest. If two paths resolve
    # to the same physical file (e.g. symlink aliases), keep one; otherwise stop
    # and print every conflicting path rather than silently choosing one.
    by_id = {}
    duplicate_paths = {}
    deduped_t1_files = []
    for t1 in t1_files:
        sid = subject_id_from_t1(t1)
        if not sid:
            print(f'[SKIP_BAD_T1_NAME] {t1}')
            continue
        resolved = t1.resolve()
        if sid not in by_id:
            by_id[sid] = (resolved, t1)
            deduped_t1_files.append(t1)
        elif by_id[sid][0] == resolved:
            print(f'[DUPLICATE_ALIAS_IGNORED] {sid}: {t1}')
        else:
            duplicate_paths.setdefault(sid, [by_id[sid][1]]).append(t1)

    if duplicate_paths:
        print('ERROR: Multiple different T1 files found for the same session ID.')
        for sid, paths in sorted(duplicate_paths.items()):
            print(f'  {sid}')
            for path in paths:
                print(f'    {path}')
        raise SystemExit(
            'Resolve duplicate T1 inputs before manifest generation. '
            'No new manifest was written.'
        )

    t1_files = deduped_t1_files
    rows = []
    for ix, t1 in enumerate(t1_files, 1):
        sid = subject_id_from_t1(t1)
        if not sid:
            continue

        lesion = lesion_index.get(sid)
        mni_shape = ''
        mni_voxel = ''
        mni_orientation = ''
        if sid in duplicate_lesions:
            mode, lesion_path, lesion_status = 'REVIEW_REQUIRED', '', 'DUPLICATE_MNI_LESION_FILES'
        elif lesion is None:
            mode, lesion_path, lesion_status = 'NO_LESION', '', 'MNI_LESION_NOT_FOUND'
        else:
            try:
                li = nib.load(str(lesion))
                if len(li.shape) != 3:
                    raise ValueError(f'not 3D: shape={li.shape}')
                arr = np.asanyarray(li.dataobj)
                mni_shape = str(tuple(int(x) for x in li.shape[:3]))
                mni_voxel = str(tuple(float(x) for x in li.header.get_zooms()[:3]))
                mni_orientation = ''.join(nib.aff2axcodes(li.affine))
                if not np.isfinite(arr).all():
                    raise ValueError('mask contains NaN/Inf')
                nvox = int(np.count_nonzero(arr > 0))
                if nvox == 0:
                    mode, lesion_path, lesion_status = 'REVIEW_REQUIRED', '', 'MNI_LESION_EMPTY'
                else:
                    mode, lesion_path, lesion_status = 'MNI_MASK', str(lesion), 'MNI_LESION_FOUND'
            except Exception as e:
                mode, lesion_path, lesion_status = 'REVIEW_REQUIRED', '', f'MNI_LESION_READ_ERROR:{type(e).__name__}'

        try:
            ti = nib.load(str(t1))
            if len(ti.shape) != 3:
                raise ValueError(f'T1 is not 3D: {ti.shape}')
            vox = tuple(float(x) for x in ti.header.get_zooms()[:3])
            shape = tuple(int(x) for x in ti.shape[:3])
            orient = ''.join(nib.aff2axcodes(ti.affine))
        except Exception as e:
            raise SystemExit(f'T1 header error for {sid}: {e}')

        group = clinical.get(sid, 'UNVERIFIED')
        rows.append({
            'id': sid,
            'code': sid,
            't1': str(t1),
            'lesion': lesion_path,
            'lesion_mode': mode,
            'lesion_status': lesion_status,
            'group': group,
            'group_verified': 'YES' if group != 'UNVERIFIED' else 'NO',
            't1_shape': str(shape),
            't1_voxel_mm': str(vox),
            't1_orientation': orient,
            'mni_lesion_shape': mni_shape,
            'mni_lesion_voxel_mm': mni_voxel,
            'mni_lesion_orientation': mni_orientation,
        })
        print(f'[{ix}/{len(t1_files)}] {sid}: {mode}; group={group}; T1={vox}')

    if not rows:
        raise SystemExit('No valid TMS T1 rows found.')

    md = out_root / '00_manifest'
    md.mkdir(parents=True, exist_ok=True)
    manifest = md / 'manifest_TMS_ses01.csv'
    report = md / 'manifest_TMS_ses01_report.csv'
    fields = list(rows[0].keys())
    with manifest.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with report.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    for name, selected in [
        ('subjects_MNI_lesion.txt', [r['id'] for r in rows if r['lesion_mode'] == 'MNI_MASK']),
        ('subjects_NO_LESION.txt', [r['id'] for r in rows if r['lesion_mode'] == 'NO_LESION']),
        ('subjects_REVIEW_REQUIRED.txt', [r['id'] for r in rows if r['lesion_mode'] == 'REVIEW_REQUIRED']),
    ]:
        (md / name).write_text('\n'.join(selected) + ('\n' if selected else ''), encoding='utf-8')

    counts = Counter(r['lesion_mode'] for r in rows)
    print('\n========== TMS MANIFEST SUMMARY ==========')
    print('T1 count:', len(rows))
    print('MNI_MASK:', counts['MNI_MASK'])
    print('NO_LESION:', counts['NO_LESION'])
    print('REVIEW_REQUIRED:', counts['REVIEW_REQUIRED'])
    print('Clinical group verified:', sum(r['group_verified'] == 'YES' for r in rows))
    print('Clinical group unverified:', sum(r['group_verified'] == 'NO' for r in rows))
    print('MANIFEST:', manifest)
    print('REPORT:', report)
    if len(rows) != 107 or counts['MNI_MASK'] != 66 or counts['NO_LESION'] != 41:
        print('WARNING: counts differ from previously reported 107 / 66 / 41. Inspect manifest; do not assume this is an error.')
    if counts['REVIEW_REQUIRED']:
        print('WARNING: resolve REVIEW_REQUIRED subjects before running the cohort.')
    if not a.clinical_table:
        print('WARNING: group column is UNVERIFIED. Confirm all NO_LESION subjects are controls using the clinical table.')

if __name__ == '__main__':
    main()
