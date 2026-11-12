#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure NumPy/Pandas helpers for a lesion-masked BN246 R2SN.

Original R2SN normalization: min-max per feature across ROI rows, then
Pearson correlation between the 22-dimensional regional feature vectors.
This module uses that exact formula when all ROI rows are valid. Fully
excluded ROIs are not entered in min-max normalization and their edges
are encoded 0, never as an artificial constant feature vector.
Uncomputable, NONEMPTY ROI edges are NaN (and separately flagged), not 0.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

NROI = 246


def classify_rois(atlas, lesion, excluded_mask, nroi=NROI, min_voxels=200):
    """Return clean atlas and list of per-ROI records; no lesion% rejection.

    `lesion` means actual segmented lesion; `excluded_mask` may additionally
    include an optional margin. A mask-erased ROI and a true 100% lesion are
    distinguished in the QC metadata.
    """
    if atlas.shape != lesion.shape or atlas.shape != excluded_mask.shape:
        raise ValueError('atlas/lesion/excluded_mask shape mismatch')
    atlas_mask = (atlas >= 1) & (atlas <= nroi)
    labels = atlas[atlas_mask].astype(np.int32, copy=False)
    total_counts = np.bincount(labels, minlength=nroi+1)
    lesion_counts = np.bincount(labels[lesion[atlas_mask]], minlength=nroi+1)
    excluded_counts = np.bincount(labels[excluded_mask[atlas_mask]], minlength=nroi+1)
    clean = np.where(atlas_mask & (~excluded_mask), atlas, 0).astype(np.int16)
    records = []
    for i in range(1, nroi + 1):
        n_total = int(total_counts[i])
        n_lesion = int(lesion_counts[i])
        n_excluded = int(excluded_counts[i])
        n_left = n_total-n_excluded
        frac = n_lesion / n_total if n_total else float('nan')
        if n_total == 0:
            status = 'ATLAS_MISSING'
        elif n_lesion == n_total:
            status = 'FULLY_LESIONED_ZERO'
        elif n_left == 0:
            status = 'EMPTY_AFTER_MARGIN_ZERO'
        elif n_left < min_voxels:
            status = 'LOW_VOXELS_ATTEMPT'
        else:
            status = 'EXTRACT_PENDING'
        records.append(dict(
            ROI=i, total_voxels=n_total, lesion_voxels=n_lesion,
            lesion_fraction=frac, excluded_voxels=n_excluded,
            clean_voxels=n_left, clean_fraction=(n_left/n_total if n_total else float('nan')),
            status=status))
    return clean, records


def valid_feature_network(feature_df, picked, qc_rows, nroi=NROI, eps=1e-12):
    """Return (W, node_valid_mask, feature_matrix, details).

    W[zero ROI, :] = W[:, zero ROI] = 0, including diagonal.
    W[nonempty but failed extraction, :] = NaN, including diagonal.
    Remaining nodes use original R2SN min-max features and Pearson r.
    """
    if len(qc_rows) != nroi or [int(r['ROI']) for r in qc_rows] != list(range(1, nroi+1)):
        raise ValueError('QC rows must contain ROI labels 1..N in order')
    if len(set(picked)) != len(picked) or len(picked) < 2:
        raise ValueError('Invalid selected feature names')
    for k in ['Label'] + picked:
        if k not in feature_df.columns and len(feature_df):
            raise ValueError('Missing feature column: ' + k)
    if len(feature_df):
        labels = pd.to_numeric(feature_df['Label'], errors='coerce')
        if labels.isna().any():
            raise ValueError('Non-numeric feature label found')
        labels = labels.astype(int)
        if labels.duplicated().any() or not labels.between(1,nroi).all():
            raise ValueError('Duplicate or out-of-range feature label')
        feature_df = feature_df.copy()
        feature_df['Label'] = labels
        feature_df = feature_df.set_index('Label')
    else:
        feature_df = pd.DataFrame(index=pd.Index([],name='Label'))
    full_features = np.full((nroi, len(picked)), np.nan, dtype=np.float64)
    node_status = []
    empty = {'FULLY_LESIONED_ZERO', 'EMPTY_AFTER_MARGIN_ZERO'}
    for idx, r in enumerate(qc_rows):
        old = r['status']
        if old in empty:
            node_status.append(old)
            continue
        if old in ('ATLAS_MISSING', 'NO_BRAIN_COVERAGE_MISSING'):
            node_status.append(old)
            continue
        label = idx+1
        if label not in feature_df.index:
            node_status.append('FEATURE_EXTRACTION_FAILED')
            continue
        vals = pd.to_numeric(feature_df.loc[label, picked], errors='coerce').to_numpy(dtype=float)
        if np.any(~np.isfinite(vals)):
            node_status.append('INVALID_FEATURE_VALUES')
            continue
        full_features[idx] = vals
        node_status.append('LOW_VOXELS_EXTRACTED' if old == 'LOW_VOXELS_ATTEMPT' else 'VALID')
    candidate_idx = np.array([i for i,s in enumerate(node_status) if s in ('VALID','LOW_VOXELS_EXTRACTED')], dtype=int)
    W = np.full((nroi,nroi), np.nan, dtype=np.float64)
    z_idx = [i for i,s in enumerate(node_status) if s in empty]
    if z_idx:
        W[z_idx,:] = 0.0
        W[:,z_idx] = 0.0
    used_features = []
    dropped = []
    if len(candidate_idx):
        X = full_features[candidate_idx]
        spread = X.max(axis=0)-X.min(axis=0)
        keep = np.isfinite(spread) & (spread > eps)
        used_features = [p for p,k in zip(picked,keep) if k]
        dropped = [p for p,k in zip(picked,keep) if not k]
        if keep.sum() >= 2:
            X = (X[:,keep] - X[:,keep].min(axis=0))/(X[:,keep].max(axis=0)-X[:,keep].min(axis=0))
            centered = X-X.mean(axis=1,keepdims=True)
            norm = np.linalg.norm(centered,axis=1)
            good = norm > eps
            for j in candidate_idx[~good]:
                node_status[j] = 'DEGENERATE_NORMALIZED_VECTOR'
            valid_idx = candidate_idx[good]
            if len(valid_idx):
                U = centered[good] / norm[good, None]
                sub = np.clip(U @ U.T, -1.0, 1.0)
                sub = (sub + sub.T)/2.0
                np.fill_diagonal(sub,1.0)
                W[np.ix_(valid_idx,valid_idx)] = sub
                # Edges from *zero* ROI must remain zero, including to valid nodes.
        else:
            for j in candidate_idx:
                node_status[j] = 'INSUFFICIENT_VARIABLE_FEATURES'
    valid_mask = np.array([s in ('VALID','LOW_VOXELS_EXTRACTED') for s in node_status],dtype=bool)
    details = dict(n_valid=int(valid_mask.sum()), n_zero=len(z_idx),
                   n_other_missing=int(nroi-valid_mask.sum()-len(z_idx)),
                   n_low_voxels_extracted=sum(s=='LOW_VOXELS_EXTRACTED' for s in node_status),
                   selected_features=len(picked),used_features=len(used_features),
                   dropped_constant_features=dropped,
                   zero_rois=[int(i+1) for i in z_idx],
                   missing_rois=[int(i+1) for i,s in enumerate(node_status) if not valid_mask[i] and i not in z_idx])
    return W,valid_mask,full_features,node_status,details


def nodal_metrics(W, valid_mask, qc_rows, statuses):
    rows=[]
    for i in range(W.shape[0]):
        a=W[i].copy(); a[i]=np.nan
        defined=np.isfinite(a)
        vals=a[defined]
        active_mask=valid_mask.copy();active_mask[i]=False
        active=a[active_mask & np.isfinite(a)]
        zstatus = statuses[i] in ('FULLY_LESIONED_ZERO','EMPTY_AFTER_MARGIN_ZERO')
        rows.append(dict(ROI=i+1,node_status=statuses[i],
                         lesion_fraction=qc_rows[i]['lesion_fraction'],
                         clean_voxels=qc_rows[i]['clean_voxels'],
                         n_defined_edges=int(defined.sum()),
                         n_active_edges=int(len(active)),
                         mean_similarity_all_defined=float(vals.mean()) if len(vals) else float('nan'),
                         mean_similarity_active=float(active.mean()) if len(active) else float('nan'),
                         positive_strength_all_defined=float(np.clip(vals,0,None).sum()) if len(vals) else float('nan'),
                         negative_abs_strength_all_defined=float(np.clip(-vals,0,None).sum()) if len(vals) else float('nan'),
                         is_zero_coded=int(zstatus)))
    return rows


def atomic_save_npy(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,arr)
    tmp.replace(path)


def write_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        if not rows:raise ValueError('Cannot determine fields for empty rows')
        fields=list(rows[0].keys())
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)
