TMS 队列 BN246 R2SN v6（基于 C3 v5 原始代码修改）
====================================================

一、这版具体做什么
------------------
对每位 TMS 被试执行同一套：原始 T1 -> SynthStrip 去颅骨 -> N4 bias correction -> ANTs 配准到 R2SN 的 MNI152 1mm 参考空间 -> BN246 ROI 特征提取 -> 原 R2SN 特征归一化与 Pearson 结构相似性网络。

两类被试的病灶处理不同：
1. lesion_mode=MNI_MASK：读取对应的 MNI lesion mask，按最近邻插值对齐到 R2SN target 网格；配准时可从 fixed metric mask 排除病灶，ROI 提取时从 BN246 ROI 中排除病灶体素。
2. lesion_mode=NO_LESION：不读取病灶文件，在 R2SN target 网格创建全零 lesion mask；不排除任何病灶体素，正常使用患者脑 mask 与 BN246 ROI 相交后提取特征。
3. lesion_mode=REVIEW_REQUIRED：manifest 会列出，但 run_parallel 会拒绝启动，必须先人工处理。

保留原 v5 的核心特征提取和网络构建函数 network_utils.py、R2SN feature_extract、22 个 picked features、min-max across valid ROI features、Pearson 相似性计算。

完全病灶损毁 ROI：零行/零列。非空但特征提取失败、患者脑覆盖不足：NaN/缺失节点，不当作病灶零节点。

重要：NO_LESION 代表没有找到 MNI 病灶文件，不等于已由临床资料证实为健康对照。正式统计前必须核对临床分组，防止将病灶文件缺失的卒中患者误分到健康组。

二、输入路径（按目前对话中提供的 TMS 路径）
--------------------------------------------
T1：
/data02/linzhenzong/TMS-MRI/Data_flip
预期格式：
Data_flip/sub-TMS001ses01/anat/sub-TMS001ses01_T1w.nii.gz

MNI lesion：
/data02/linzhenzong/TMS-MRI/Lesion/lesion_mask_MNI1mm_flip
预期格式：
sub-TMS001ses01_space-MNI152NLin6Asym_res-01_label-lesion_mask.nii.gz

输出根目录：
/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2

T1 原始文件不会复制、覆盖或修改。

三、安装/环境检查
-----------------
conda activate r2sn
which mri_synthstrip
python -c "import R2SN,ants,nibabel,scipy,pandas,radiomics,matplotlib; print('ALL_IMPORTS_OK')"

如果 mri_synthstrip 不存在，先配置 FreeSurfer/SynthStrip；不要用粗糙阈值法替代。

四、部署代码
------------
将 ZIP 上传至 4090，然后：

mkdir -p /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/scripts
cd /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/scripts
unzip -o /你的上传路径/R2SN_TMS_BN246_T1prep_4090_v6.zip
cd R2SN_TMS_BN246_T1prep_4090_v6

python -m compileall -q .
python selftest.py
bash check_environment.sh

selftest 预期：
PASS: no-lesion control, surviving ROI, 100% destroyed ROI -> zero, nonempty failed -> NaN, official Pearson equivalence

五、生成 TMS manifest
--------------------
python make_manifest_tms.py --pattern 'sub-TMS*ses01/anat/*_T1w.nii.gz'

预期此前观察到的数量：T1=107、MNI_MASK=66、NO_LESION=41、REVIEW_REQUIRED=0。
数量不一致时先查看：
/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/00_manifest/manifest_TMS_ses01_report.csv

可选：若已有临床 CSV，并且列名为 subject_id、group：
python make_manifest_tms.py --clinical-table /path/to/clinical.csv

如列名不同：
python make_manifest_tms.py --clinical-table /path/to/clinical.csv \
  --clinical-id-col subject_id --clinical-group-col group

没有提供 clinical table 时，group=UNVERIFIED。算法仍可按 lesion_mode 执行，但统计分组前必须合并临床表。

六、准备 R2SN 的共享 BN246/MNI 参考模板（只需一次）
---------------------------------------------------
python prepare_reference.py --synthstrip-cmd /data01/home/tanghui/lesion_seg_and_MNI/models/synthstrip/mri_synthstrip

这一步沿用 R2SN 包内的：
- MNI152_T1_1mm.nii
- rBN_Atlas_246_1mm.nii
并调用 SynthStrip 为模板生成同网格 brain mask。不要替换 atlas 或 target。

七、先做 2 名试运行：1 名卒中 + 1 名健康对照
-------------------------------------------
STROKE_ID=$(head -n 1 /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/00_manifest/subjects_MNI_lesion.txt)
CONTROL_ID=$(head -n 1 /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/00_manifest/subjects_NO_LESION.txt)
echo "$STROKE_ID"
echo "$CONTROL_ID"

先运行卒中病例：
python -u run_parallel.py --id "$STROKE_ID" \
  --workers 1 --threads-per-worker 4 \
  --keep-qc-images --keep-transforms \
  2>&1 | tee stroke_pilot_v6.log

再运行无病灶病例：
python -u run_parallel.py --id "$CONTROL_ID" \
  --workers 1 --threads-per-worker 4 \
  --keep-qc-images --keep-transforms \
  2>&1 | tee control_pilot_v6.log

默认 margin-mm=0，即只排除病灶真实覆盖体素，不额外膨胀病灶。
默认 transform=SyN，registration-mask=fixed，min-voxels=200。
TMS T1 如为 1x1x1 mm，不需要 --allow-thick-t1。

八、查看试运行结果
------------------
OUT=/data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/subjects
TAG=TMS_BN246_T1prep_v6

python verify_v5.py --id "$STROKE_ID" --root "$OUT" --tag "$TAG"
python inspect_network.py --id "$STROKE_ID" --root "$OUT" --tag "$TAG" --plot

python verify_v5.py --id "$CONTROL_ID" --root "$OUT" --tag "$TAG"
python inspect_network.py --id "$CONTROL_ID" --root "$OUT" --tag "$TAG" --plot

QC 位置：
$OUT/<subject_id>/TMS_BN246_T1prep_v6/qc/
重点检查：
- T1prep_lesion_overlay.png
- registered_T1_MNI.nii.gz
- patient_brain_mask_MNI.nii.gz
- lesion_resampled_MNI.nii.gz
- BN_lesionExcluded.nii.gz
- ROI_lesion_overlap.csv
- run_summary.json

健康对照的 lesion_resampled_MNI.nii.gz 应为全零图像；BN_lesionExcluded.nii.gz 应只由患者脑 mask 覆盖范围内的 BN246 ROI 组成，不应因病灶而删除体素。
卒中病例应确认 MNI lesion 与配准 T1 解剖位置吻合。尤其注意 MNI152NLin6Asym 与 R2SN 自带 MNI 模板的空间定义；重采样只改变网格，不会修正两个模板之间的解剖空间差异。必须人工查看叠加 QC。

九、通过 QC 后扩大样本
--------------------
建议先 4-10 例，包含不同病灶位置、左右侧及健康对照：
python -u run_parallel.py --limit 6 --workers 2 --threads-per-worker 2 \
  --keep-qc-images \
  2>&1 | tee pilot6_v6.log

注意 --limit 6 是 manifest 当前顺序的前 6 名，不保证同时包含两组；正式抽样建议分别指定 --id，或人工制作 pilot manifest。

确认流程稳定后全队列运行（4090 非 Slurm，使用 Python 进程池并行）：
nohup python -u run_parallel.py \
  --workers 2 --threads-per-worker 2 \
  > /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/TMS_BN246_v6_batch.log 2>&1 < /dev/null &

tail -f /data02/tanghui/TMS-MRI/R2SN_BN246_lesionaware_v2/TMS_BN246_v6_batch.log

完成状态：
python check_status.py

并行建议：先 workers=2, threads-per-worker=2。观察 CPU、RAM 和单例耗时后再考虑 workers=3。不要一开始就将进程数开到最大，ANTs 配准和 PyRadiomics 会占用较多内存。

十、结果结构
------------
R2SN_BN246_lesionaware_v2/
  00_manifest/
  scripts/R2SN_TMS_BN246_T1prep_4090_v6/
    reference_bn246/
  subjects/
    sub-TMS001ses01/TMS_BN246_T1prep_v6/
      qc/
      radiomics/
      network/

每名被试的主要结果：
- radiomics/ROI_features_246.csv
- radiomics/ROI_node_metrics.csv
- network/<ID>_network.npy
- network/<ID>_valid_roi_mask.npy
- network/<ID>_network_zeroFilled_DISPLAY_ONLY.npy（仅展示，不用于统计）

网络主文件保留 NaN 缺失节点。不要把 DISPLAY_ONLY 的 NaN->0 文件用于正式统计。

十一、科学解释与限制
-------------------
1. 对 stroke：ROI_clean = BN246 ROI ∩ patient brain mask ∩ 非病灶体素。
2. 对 control：ROI_clean = BN246 ROI ∩ patient brain mask；病灶 mask 为全零。
3. 两组使用同一套 R2SN 特征提取参数与网络构建流程。
4. 部分病灶 ROI 用剩余体素提取；完全被病灶覆盖的 ROI 零编码；脑覆盖不足或特征提取失败的非空 ROI 记为缺失，而不是零。
5. 两组 ROI 有效体素数量不同可能造成特征稳定性差异；应保存 clean_voxels、lesion_fraction，并在统计模型中进行敏感性分析。
6. 病灶 MNI 与 R2SN target 必须处于兼容的 MNI 世界坐标空间。最近邻重采样不是空间配准，空间不一致不能靠 resample 修复。
7. SynthStrip + N4 + ANTs 配准不等于保证所有病例解剖配准正确。必须按病灶位置、病灶侧别、脑覆盖和 Dice 进行 QC。


十二、v6.1 修正
---------------
- 默认 manifest pattern 已限制为 ses01：sub-TMS*ses01/anat/*_T1w.nii.gz
- duplicate T1 先进行完整预检；若同一 session ID 存在不同文件，会列出全部路径并停止，不会写出不完整 manifest。
- SynthStrip 默认改为本服务器已有的离线程序：
  /data01/home/tanghui/lesion_seg_and_MNI/models/synthstrip/mri_synthstrip
- prepare_reference.py 与 run_parallel.py 均可用 --synthstrip-cmd 覆盖默认路径。
- 仍须检查 ses01 的真实 T1 数量、MNI lesion 数量，并核对临床组别。
