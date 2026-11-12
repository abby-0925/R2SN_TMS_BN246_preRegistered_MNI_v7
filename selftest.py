#!/usr/bin/env python3
"""No MRI/ANTs required. Validate normal, partially damaged, zero and missing ROI logic."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from network_utils import classify_rois, valid_feature_network, atomic_save_npy


def main():
    # Simplified atlas for ROI=1..4 with 10 voxels each
    atlas = np.repeat(np.arange(1,5,dtype=np.int16),10).reshape(4,10,1)
    lesion=np.zeros(atlas.shape,dtype=bool)
    lesion[0,0,0]=True   # partially damaged ROI1: 9 remaining
    lesion[1,:,:]=True   # completely damaged ROI2
    # ROI3 full, ROI4 artificially excluded from feature csv (computational missing)
    clean,rows=classify_rois(atlas,lesion,lesion,nroi=4,min_voxels=10)
    assert rows[0]['status']=='LOW_VOXELS_ATTEMPT'
    assert rows[1]['status']=='FULLY_LESIONED_ZERO'
    assert rows[2]['status']=='EXTRACT_PENDING'
    picked=['a','b','c','d','e']
    # label 4 missing despite remaining voxels -> NaN, not biological zero
    df=pd.DataFrame({'Label':[1,3],
                     'a':[1.,3.],'b':[2.,1.],'c':[3.,2.],
                     'd':[4.,1.],'e':[5.,3.]})
    W,valid,feat,status,info=valid_feature_network(df,picked,rows,4)
    assert W.shape==(4,4)
    assert np.all(W[1]==0) and np.all(W[:,1]==0)
    assert np.isnan(W[3,0]) and np.isnan(W[0,3])
    assert np.allclose(np.diag(W)[[0,2]],[1.,1.])
    assert np.allclose(W,W.T,equal_nan=True)
    assert info['n_zero']==1 and info['n_other_missing']==1
    assert status[3]=='FEATURE_EXTRACTION_FAILED'
    # No-lesion (healthy-control) branch: no ROI is zero-coded by the lesion mask.
    no_lesion=np.zeros(atlas.shape,dtype=bool)
    clean_h,rows_h=classify_rois(atlas,no_lesion,no_lesion,nroi=4,min_voxels=1)
    assert np.all(clean_h==atlas)
    assert all(r['lesion_voxels']==0 for r in rows_h)
    assert all(r['status']=='EXTRACT_PENDING' for r in rows_h)
    # Check numerics match official min-max + row Pearson when no nodes lost
    rng=np.random.default_rng(2026)
    X=rng.normal(size=(4,5))
    df2=pd.DataFrame(X,columns=picked);df2.insert(0,'Label',range(1,5))
    r2=[dict(ROI=i,status='EXTRACT_PENDING') for i in range(1,5)]
    full,mask,_,st,_=valid_feature_network(df2,picked,r2,4)
    y=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    expected=np.corrcoef(y)
    assert np.allclose(full,expected,atol=1e-12),np.max(abs(full-expected))
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'test.npy';atomic_save_npy(f,W)
        assert np.allclose(np.load(f),W,equal_nan=True)
    print('PASS: no-lesion control, surviving ROI, 100% destroyed ROI -> zero, nonempty failed -> NaN, official Pearson equivalence')

if __name__=='__main__':main()
