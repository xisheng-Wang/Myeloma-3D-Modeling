# utils.py - 通用工具函数
import nibabel as nib
import numpy as np
import os
from scipy import ndimage

def load_ct_data(ct_path):
    """加载CT数据"""
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT文件未找到: {ct_path}")
    
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata().astype(np.float32)
    ct_affine = ct_img.affine
    
    # 计算体素物理间距
    spacing = (
        np.abs(ct_affine[0, 0]),
        np.abs(ct_affine[1, 1]),
        np.abs(ct_affine[2, 2])
    )
    
    return ct_data, ct_affine, spacing

def load_and_align_data(source_path, ref_shape, ref_affine, interpolation=1):
    """加载并配准数据到参考空间"""
    if not os.path.exists(source_path):
        return None
    
    img = nib.load(source_path)
    source_data = img.get_fdata()
    source_affine = img.affine
    
    # 计算变换矩阵
    transform_matrix = np.linalg.inv(source_affine).dot(ref_affine)
    matrix = transform_matrix[:3, :3]
    offset = transform_matrix[:3, 3]
    
    # 重采样
    resampled = ndimage.affine_transform(
        source_data,
        matrix=matrix,
        offset=offset,
        output_shape=ref_shape,
        order=interpolation,
        mode='constant',
        cval=0.0
    )
    
    return resampled