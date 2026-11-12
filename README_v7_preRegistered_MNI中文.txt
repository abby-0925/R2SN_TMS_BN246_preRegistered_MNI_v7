TMS BN246 R2SN v7：已预配准 MNI 输入模式
======================================

适用输入（同一被试、同一 MNI 1mm 网格）：
1. T1_EasyReg_MNI_brain.nii.gz       已去颅骨、已配准到 MNI 的 T1
2. T1_EasyReg_MNI_brainmask.nii.gz  与 T1 完全同网格的脑掩膜
3. lesion_MNI_on_T1grid.nii.gz      与 T1 完全同网格的病灶掩膜

重要区别：
- run_one_preregistered.py 不运行 SynthStrip。
- 不运行 ANTs Affine/SyN，不再次配准。
- 不对 T1、脑掩膜或病灶做重采样。
- 强制检查三者与 R2SN 自带 MNI target、BN246 atlas 的形状和 affine。
- 默认在现有 MNI 网格内用 brain mask 做 N4 bias correction，然后按脑内 p95 归一化到 255；该步骤不改变空间位置。
- 继续调用原 R2SN feature_extract、Params.yaml、picked_features.txt 和 network_utils.py，保留原始 22 个特征和 Pearson 相似性网络计算。

第一步：部署
------------
把本目录中的新增文件放到原 v6.1 scripts 目录（本 ZIP 已包含原文件和新增文件）：

  run_one_preregistered.py
  qc_preregistered_MNI.py

保持 run_one.py、network_utils.py、R2SN 环境不变。

第二步：激活环境并确定路径
--------------------------
conda activate r2sn

建议先设置：

OUT='/实际存放三个 MNI 文件的目录'
T1="$OUT/T1_EasyReg_MNI_brain.nii.gz"
BM="$OUT/T1_EasyReg_MNI_brainmask.nii.gz"
LESION="$OUT/lesion_MNI_on_T1grid.nii.gz"

第三步：取得 R2SN target 和 BN246 atlas 路径
-------------------------------------------
python - <<'PY'
import R2SN
from pathlib import Path
r=Path(R2SN.__file__).resolve().parent
print('TARGET=',r/'data'/'MNI152_T1_1mm.nii')
print('ATLAS =',r/'data'/'rBN_Atlas_246_1mm.nii')
print('PARAMS=',r/'network_feature'/'Params.yaml')
print('PICKED=',r/'network_feature'/'picked_features.txt')
PY

第四步：严格空间 QC（必须先通过）
----------------------------------
TARGET='/上一步打印的/MNI152_T1_1mm.nii'
ATLAS='/上一步打印的/rBN_Atlas_246_1mm.nii'

python -u qc_preregistered_MNI.py \
  --t1 "$T1" \
  --brainmask "$BM" \
  --lesion "$LESION" \
  --target "$TARGET" \
  --atlas "$ATLAS" \
  --outdir "$OUT/qc_preR2SN" \
  --min-voxels 200

只有 T1、brainmask、lesion、target、atlas 全部同网格，且 overlay 显示病灶与解剖位置吻合，才继续。若报 grid mismatch，不要通过增大 tolerance 或随意重采样强行运行；先确认 EasyReg 使用的 MNI 模板是否与 R2SN target 相同。

第五步：单例 R2SN 试运行
------------------------
假设该病例为有病灶的 sub-TMS002ses01：

SCRIPT='/实际脚本目录/R2SN_TMS_BN246_T1prep_4090_v6_1'
OUTROOT='/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects'

python -u "$SCRIPT/run_one_preregistered.py" \
  --id sub-TMS002ses01 \
  --code sub-TMS002ses01 \
  --t1 "$T1" \
  --brain-mask "$BM" \
  --lesion "$LESION" \
  --lesion-mode MNI_MASK \
  --out-root "$OUTROOT" \
  --tag TMS_BN246_preRegistered_v7 \
  --margin-mm 0 \
  --min-voxels 200 \
  --keep-qc-images \
  2>&1 | tee "$OUT/preRegistered_v7_pilot.log"

注意：这一步不再需要 prepare_reference.py，不需要 SynthStrip，不需要 --transform、--registration-mask、--reference-dir 或 --keep-transforms。

如果 N4 报错，可先检查 ants/nibabel 与输入图像；只有在确认 EasyReg T1 已做 bias correction 时，才考虑增加 --skip-n4 进行敏感性试验。不要在同一分析批次混用 N4 与非 N4 版本。

第六步：查看结果
----------------
结果目录：
$OUTROOT/sub-TMS002ses01/TMS_BN246_preRegistered_v7/

重点：
qc/run_summary.json
qc/T1prep_lesion_overlay.png
qc/registered_T1_MNI.nii.gz
qc/patient_brain_mask_MNI.nii.gz
qc/lesion_resampled_MNI.nii.gz
qc/BN_lesionExcluded.nii.gz
qc/ROI_lesion_overlap.csv
radiomics/ROI_features_246.csv
radiomics/ROI_node_metrics.csv
network/sub-TMS002ses01_network.npy
network/sub-TMS002ses01_valid_roi_mask.npy

主网络文件中的 NaN 表示非空 ROI 特征提取失败/节点缺失；完全病灶损毁 ROI 的行列为 0。*_zeroFilled_DISPLAY_ONLY.npy 只用于展示，不能用于统计。

第七步：健康对照
----------------
只有临床资料确认是健康对照后才使用 NO_LESION。命令中不传 --lesion：

  --lesion-mode NO_LESION

其余输入仍然需要对应的预配准 MNI T1 和 brainmask。不能仅因为没有 lesion 文件就将被试判定为健康对照。

第八步：批量运行前的限制
------------------------
目前先只运行一个病例。至少确认：
1. grid QC 全部通过；
2. T1/病灶 overlay 解剖位置正确；
3. N4 后 T1 没有异常强度或空洞；
4. ROI 特征成功提取；
5. 网络为 246x246，NaN/零节点数与 ROI QC 表一致。

通过后再为每位被试生成同格式的 MNI T1、brainmask、lesion，并建立新 manifest/并行脚本。不要把旧 v6 native-T1 pipeline 与 v7 pre-registered pipeline 的结果混在一起而不标记版本。
