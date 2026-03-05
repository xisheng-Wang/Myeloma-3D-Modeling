import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv
import os

class BoneExtractorAI:
    """AI增强版本骨骼提取算法 (分层策略：精确标签 + 高密度后备)"""
    
    def __init__(self, config=None):
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.AI_AVAILABLE = True
        except ImportError:
            print("TotalSegmentator未安装，请运行: pip install TotalSegmentator")
            self.AI_AVAILABLE = False
        
        self.config = {
            "use_high_resolution": True,
            "ai_task": "total",
            # 分层参数
            "high_density_threshold": 200,    # 后备高密度阈值
            # 原有迟滞阈值
            "bone_high_threshold": 100,
            "bone_low_threshold": 50,
            "gaussian_sigma": 0.5,
            "min_component_size": 500,
            "closing_radius": 1,
            "smooth_iter": 80,
            "smooth_relax": 0.04,
            "decimate_ratio": 0.1,
            "bone_color": "#f5f5f5"
        }
        if config: self.config.update(config)
    
    def run_ai_segmentation(self, ct_data, ct_affine, output_dir):
        import nibabel as nib
        os.makedirs(output_dir, exist_ok=True)
        temp_ct_path = os.path.join(output_dir, "temp_ct.nii.gz")
        ai_output_path = os.path.join(output_dir, "ai_segmentation_v2.nii.gz")
        
        if os.path.exists(ai_output_path):
            return ai_output_path
        
        t0 = time.time()
        try:
            from totalsegmentator.python_api import totalsegmentator
            nib.save(nib.Nifti1Image(ct_data, ct_affine), temp_ct_path)
            totalsegmentator(input=temp_ct_path, output=ai_output_path, fast=not self.config['use_high_resolution'], ml=True, task=self.config['ai_task'])
            if os.path.exists(temp_ct_path): os.remove(temp_ct_path)
            return ai_output_path
        except Exception as e:
            if os.path.exists(temp_ct_path): os.remove(temp_ct_path)
            raise e
    
    def extract_bone(self, ct_data, ct_affine, spacing, ai_output_path):
        t0 = time.time()
        import nibabel as nib
        ai_data = nib.load(ai_output_path).get_fdata()
        
        # ========== 分层策略 ==========
        # 第一层：精确骨骼标签
        V2_ALL_BONES = list(range(25, 51)) + list(range(69, 79)) + [91] + list(range(92, 118))
        precise_bone = np.isin(ai_data, V2_ALL_BONES)
        
        # 第二层：高密度后备（在非精确标签区域中寻找高CT值）
        non_precise = ~precise_bone
        ct_smooth = ndimage.gaussian_filter(ct_data, sigma=self.config['gaussian_sigma'])
        high_density = (ct_smooth > self.config["high_density_threshold"]) & non_precise
        
        # 合并得到最终骨骼区域
        combined_mask = precise_bone | high_density
        
        # ========== 原有迟滞阈值算法，但作用域改为 combined_mask ==========
        mask_high = (ct_smooth > self.config["bone_high_threshold"]) & combined_mask
        mask_low = (ct_smooth > self.config["bone_low_threshold"]) & combined_mask
        
        labeled_low, num_features = ndimage.label(mask_low)
        valid_labels = np.unique(labeled_low[mask_high])
        valid_labels = valid_labels[valid_labels != 0]
        
        bone_mask = mask_high if len(valid_labels) == 0 else np.isin(labeled_low, valid_labels)
        bone_closed = morphology.binary_closing(bone_mask, morphology.ball(self.config["closing_radius"]))
        
        labeled_final, num_final = ndimage.label(bone_closed)
        if num_final > 0:
            sizes = ndimage.sum(np.ones_like(bone_closed, dtype=int), labeled_final, range(1, num_final + 1))
            final_mask = np.isin(labeled_final, np.where(sizes >= self.config["min_component_size"])[0] + 1)
        else:
            final_mask = bone_closed
            
        return ndimage.binary_fill_holes(final_mask)
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        meshes = {}
    
        # 骨骼网格
        bone_grid = pv.wrap(bone_mask.astype(np.uint8))
        bone_grid.spacing = spacing
        bone_mesh = bone_grid.contour([0.5])
        bone_mesh = bone_mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True
        )
        bone_mesh = bone_mesh.compute_normals(point_normals=True, auto_orient_normals=True)
        if self.config["decimate_ratio"] > 0:
            bone_mesh = bone_mesh.decimate(self.config["decimate_ratio"])
        meshes['bone'] = bone_mesh

        # 骨髓瘤网格
        if label_data is not None and np.any(label_data):
            tumor_float = ndimage.gaussian_filter(label_data.astype(np.float32), sigma=0.8)
            tumor_grid = pv.wrap(tumor_float)
            tumor_grid.spacing = spacing
            tumor_mesh = tumor_grid.contour([0.5])
            tumor_mesh = tumor_mesh.smooth(n_iter=5, relaxation_factor=0.01)
            tumor_mesh = tumor_mesh.compute_normals(point_normals=True)
            meshes['tumor'] = tumor_mesh

        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        os.makedirs(output_dir, exist_ok=True)
        for mesh_name, mesh in meshes.items():
            mesh.save(os.path.join(output_dir, f"{case_name}_AI_{mesh_name}.obj"))