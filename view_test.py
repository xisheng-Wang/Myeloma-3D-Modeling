# #!/usr/bin/env python
# """
# extract_bone_from_this_version.py

# 根据当前 TotalSegmentator 版本的实际标签（打印结果）提取所有骨骼，
# 生成原始体素模型（0平滑，0过滤），保存到 AI 输出目录。
# """

# import numpy as np
# import nibabel as nib
# import pyvista as pv
# import os
# import time
# from config_paths import PathConfig
# from utils import load_ct_data

# # ===== 当前版本正确的骨骼标签列表（基于你打印的结果） =====
# BONE_LABELS = (
#     [25] +                     # 骶骨
#     list(range(26, 51)) +      # 脊椎 S1~C1 (26~50)
#     [69, 70, 71, 72, 73, 74, 75, 76, 77, 78] +  # 四肢骨
#     [91] +                     # 颅骨
#     list(range(92, 116)) +     # 左肋 (92~103) + 右肋 (104~115)
#     [116, 117]                 # 胸骨, 肋软骨
# )

# def run_total_segmentation(ct_data, ct_affine, output_dir, use_high_res=False):
#     from totalsegmentator.python_api import totalsegmentator
#     os.makedirs(output_dir, exist_ok=True)
#     temp_ct_path = os.path.join(output_dir, "temp_ct.nii.gz")
#     seg_output_path = os.path.join(output_dir, "total_seg.nii.gz")

#     if os.path.exists(seg_output_path):
#         print(f"检测到已有的分割结果: {seg_output_path}")
#         return seg_output_path

#     print(">>> 启动 TotalSegmentator 全身分割任务 <<<")
#     t0 = time.time()
#     temp_img = nib.Nifti1Image(ct_data, ct_affine)
#     nib.save(temp_img, temp_ct_path)

#     try:
#         totalsegmentator(
#             input=temp_ct_path,
#             output=seg_output_path,
#             fast=not use_high_res,
#             ml=True,
#             task="total",
#             preview=False,
#             force_split=False
#         )
#         print(f"✅ 分割完成! 耗时: {time.time()-t0:.2f}秒")
#         return seg_output_path
#     finally:
#         if os.path.exists(temp_ct_path):
#             os.remove(temp_ct_path)

# def create_raw_mesh(mask, spacing):
#     """原始体素网格（无平滑，仅计算法线）"""
#     grid = pv.wrap(mask.astype(np.uint8))
#     grid.spacing = spacing
#     mesh = grid.contour([0.5])
#     mesh = mesh.compute_normals(point_normals=True, auto_orient_normals=True)
#     return mesh

# def main():
#     print("\n" + "="*70)
#     print("     根据当前版本实际标签提取全部骨骼（原始体素）")
#     print("="*70)

#     case = PathConfig.SINGLE_CASE
#     case_name = case['case_name']
#     print(f"处理病例: {case_name}")

#     try:
#         ct_data, ct_affine, spacing = load_ct_data(case['ct_path'])
#     except Exception as e:
#         print(f"❌ 加载CT失败: {e}")
#         return

#     work_dir = os.path.join(PathConfig.OUTPUT_PATHS["AI_TEMP"], f"{case_name}_bone_extract")
#     os.makedirs(work_dir, exist_ok=True)

#     try:
#         seg_path = run_total_segmentation(ct_data, ct_affine, work_dir, use_high_res=False)
#         seg_img = nib.load(seg_path)
#         seg_data = seg_img.get_fdata().astype(np.int16)

#         print(f"\n正在提取骨骼掩码 (共 {len(BONE_LABELS)} 个标签)...")
#         bone_mask = np.isin(seg_data, BONE_LABELS)

#         mesh = create_raw_mesh(bone_mask, spacing)

#         out_dir = PathConfig.OUTPUT_PATHS["AI"]
#         os.makedirs(out_dir, exist_ok=True)
#         out_path = os.path.join(out_dir, f"{case_name}_ALL_BONES_RAW.obj")

#         print("\n正在写入 OBJ 文件...")
#         mesh.save(out_path)
#         print(f"✅ 骨骼模型已导出: {out_path} ({mesh.n_cells:,}面)")

#     except Exception as e:
#         print(f"❌ 处理过程中发生错误: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()
"""
view_test.py (V2 绝对正确标签验真版)
直接读取刚刚生成的 V2 分割结果，100% 提取真正的全套骨骼（含肋软骨）的原始马赛克模型。
"""

import numpy as np
import nibabel as nib
import pyvista as pv
import os
import time

from config_paths import PathConfig
from utils import load_ct_data

# === V2 版本官方全套骨骼标签 (这次绝对不会有肠子和肺了！) ===
V2_BONE_LABELS = (
    list(range(25, 51)) +    # 25: sacrum(骶骨), 26-50: 脊椎 (S1, L5-L1, T12-T1, C7-C1)
    list(range(69, 79)) +    # 69-70: 肱骨, 71-72: 肩胛骨, 73-74: 锁骨, 75-76: 股骨, 77-78: 骨盆(hip)
    [91] +                   # 91: 颅骨(skull)
    list(range(92, 116)) +   # 92-115: 左右肋骨共24根
    [116, 117]               # 116: 胸骨(sternum), 117: 肋软骨(costal_cartilages)
)

def create_raw_mesh(mask, spacing):
    """绝对原始的网格生成：原汁原味的体素方块"""
    print("\n    -> [Mesh] 正在生成原始体素网格（带有强烈的马赛克方块感）...")
    grid = pv.wrap(mask.astype(np.uint8))
    grid.spacing = spacing
    mesh = grid.contour([0.5])
    mesh = mesh.compute_normals(point_normals=True, auto_orient_normals=True)
    return mesh

def main():
    print("\n" + "="*70)
    print("     🔍 V2 极致真相版：提取 100% 纯净的升级版 AI 骨骼")
    print("="*70)

    case = PathConfig.SINGLE_CASE
    case_name = case['case_name']
    print(f"处理病例: {case_name}")

    try:
        ct_data, ct_affine, spacing = load_ct_data(case['ct_path'])
    except Exception as e:
        print(f"❌ 加载CT失败: {e}")
        return

    work_dir = os.path.join(PathConfig.OUTPUT_PATHS["AI_TEMP"], f"{case_name}_V2_test")
    seg_path = os.path.join(work_dir, "total_seg_v2.nii.gz")
    
    if not os.path.exists(seg_path):
        print(f"❌ 找不到 V2 分割文件：{seg_path}")
        print("请确保你上一步已经成功运行并生成了该文件。")
        return

    try:
        print(f"\n正在加载现有的 V2 分割数据: {seg_path}")
        t0 = time.time()
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata().astype(np.int16)
        print(f"加载完成，耗时: {time.time()-t0:.2f}秒")

        # 100% 原样提取！直接套用真正的 V2 骨骼白名单！
        print(f"\n正在提取原生骨骼掩码 (核对标签数量: {len(V2_BONE_LABELS)})...")
        bone_mask = np.isin(seg_data, V2_BONE_LABELS)

        # 生成最粗糙、真实的模型
        mesh = create_raw_mesh(bone_mask, spacing)

        out_dir = PathConfig.OUTPUT_PATHS["AI"]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{case_name}_V2_REAL_BONE.obj")
        
        print("\n正在写入 OBJ 文件，请耐心等待...")
        mesh.save(out_path)
        print(f"✅ V2 真实原始骨骼模型已导出: {out_path} ({mesh.n_cells:,}面)")

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()