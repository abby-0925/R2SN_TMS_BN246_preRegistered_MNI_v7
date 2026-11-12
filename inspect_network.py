#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect one existing TMS BN246 T1prep v6 network, without recomputation."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--id',required=True)
    p.add_argument('--root',default='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects')
    p.add_argument('--tag',default='TMS_BN246_T1prep_v6')
    p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    base=Path(a.root)/a.id/a.tag
    network=np.load(base/'network'/f'{a.id}_network.npy')
    roi_valid=np.load(base/'network'/f'{a.id}_valid_roi_mask.npy')
    summary=json.loads((base/'qc'/'run_summary.json').read_text(encoding='utf-8'))
    assert network.shape==(246,246) and roi_valid.shape==(246,)
    print('Patient:',a.id,'Shape:',network.shape)
    print('Status:',summary.get('status'),'Valid:',int(roi_valid.sum()),
          'Fully zero-coded:',summary.get('n_zero'),
          'Other missing:',summary.get('n_other_missing'))
    print('Finite:',int(np.isfinite(network).sum()),'NaNs:',int(np.isnan(network).sum()))
    print('Symmetric with NaN:',np.allclose(network,network.T,equal_nan=True))
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,7))
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('gray')
        im=ax.imshow(network,vmin=-1,vmax=1,cmap=cmap,interpolation='nearest')
        ax.set_title(f'BN246 R2SN partial ROI: {a.id}  (gray=missing, zero=destroyed)')
        ax.set_xlabel('ROI');ax.set_ylabel('ROI')
        fig.colorbar(im,ax=ax,shrink=.8)
        out=base/'network'/f'{a.id}_R2SN_heatmap.png'
        fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)
        print('Heatmap:',out)

if __name__=='__main__':main()
