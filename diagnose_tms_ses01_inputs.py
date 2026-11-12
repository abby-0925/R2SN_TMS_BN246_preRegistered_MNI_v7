#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict

root = Path("/data02/linzhenzong/TMS-MRI/Data_flip")
files = sorted(root.glob("sub-TMS*ses01/anat/*_T1w.nii.gz"))
by_id = defaultdict(list)

for p in files:
    sid = p.name.replace("_T1w.nii.gz", "")
    by_id[sid].append(p)

dups = {k:v for k,v in by_id.items() if len(v) > 1}
print("SES01 T1 files:", len(files))
print("Unique session IDs:", len(by_id))
print("Duplicate IDs:", len(dups))
for sid, paths in sorted(dups.items()):
    print("\\nDUPLICATE:", sid)
    for p in paths:
        print(" ", p)
print("\\nFirst 10 files:")
for p in files[:10]:
    print(p)
