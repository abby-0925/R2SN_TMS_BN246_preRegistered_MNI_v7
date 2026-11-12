#!/usr/bin/env bash
set -u
SYNTHSTRIP="/data01/home/tanghui/lesion_seg_and_MNI/models/synthstrip/mri_synthstrip"
if [ -x "$SYNTHSTRIP" ]; then
  echo "OK SynthStrip executable: $SYNTHSTRIP"
  "$SYNTHSTRIP" --help >/dev/null 2>&1 || true
elif [ -f "$SYNTHSTRIP" ]; then
  echo "FOUND but not executable: $SYNTHSTRIP"
  echo "Try: chmod u+x \"$SYNTHSTRIP\""
else
  echo "MISSING SynthStrip executable: $SYNTHSTRIP"
  echo "Edit --synthstrip-cmd to the actual executable path."
fi
python - <<'PY'
for m in ('R2SN','ants','nibabel','numpy','pandas','scipy','radiomics','matplotlib'):
 try:
  x=__import__(m);print('OK',m,getattr(x,'__version__',''))
 except Exception as e:print('MISSING',m,str(e))
PY
