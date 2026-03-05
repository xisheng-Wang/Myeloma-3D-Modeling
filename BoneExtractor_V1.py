import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv
import os
import tempfile
import shutil

class BoneExtractorV1:
    """V1稳健版 (分层策略 + 软骨搭桥)"""
    
    def __init__(self, config=None):
        self.AI_AVAILABLE = True
        self.config = {
            "use_high_resolution": True, 
            "ai_task": "total",
            # 分层参数
            "high_density_threshold": 200,
            # 原有参数
            "bone_core_threshold": 150, 
            "cartilage_threshold": 80,
            "dilation_iter": 2,
            "gaussian_sigma": 0.5,
            "closing_radius": 2, 
            "min_component_size": 1500,
            "mesh_smooth_sigma": 0.6, 
            "smooth_iter": 40, 
            "smooth_relax": 0.02, 
            "decimate_ratio": 0.1
        }
        if config: self.config.update(config)
    
    def run_ai_segmentation(self, ct_data, ct_affine, output_dir):
        import nibabel as nib
        temp_ct_path = os.path.join(output_dir, "temp_ct.nii.gz")
        ai_output_path = os.path.join(output_dir, "ai_segmentation_v2.nii.gz")
        if os.path.exists(ai_output_path): return ai_output_path
        
        from totalsegmentator.python_api import totalsegmentator
        nib.save(nib.Nifti1Image(ct_data, ct_affine), temp_ct_path)
        totalsegmentator(input=temp_ct_path, output=ai_output_path, fast=not self.config['use_high_resolution'], ml=True, task=self.config['ai_task'])
        if os.path.exists(temp_ct_path): os.remove(temp_ct_path)
        return ai_output_path
            
    def extract_bone(self, ct_data, spacing):
        affine = np.eye(4); affine[0,0], affine[1,1], affine[2,2] = spacing
        temp_dir = tempfile.mkdtemp(prefix="V1_AI_")
        try:
            ai_output_path = self.run_ai_segmentation(ct_data, affine, temp_dir)
            return self._extract_bone_with_ai(ct_data, affine, spacing, ai_output_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def _extract_bone_with_ai(self, ct_data, ct_affine, spacing, ai_output_path):
        import nibabel as nib
        ai_data = nib.load(ai_output_path).get_fdata()
        ct_smooth = ndimage.gaussian_filter(ct_data, sigma=self.config['gaussian_sigma'])
        
        # ========== 分层策略 ==========
        V2_ALL_BONES = list(range(25, 51)) + list(range(69, 79)) + [91] + list(range(92, 118))
        precise_bone = np.isin(ai_data, V2_ALL_BONES)
        non_precise = ~precise_bone
        high_density = (ct_smooth > self.config["high_density_threshold"]) & non_precise
        combined_mask = precise_bone | high_density
        
        # ========== 原有软骨搭桥算法，作用域改为 combined_mask ==========
        core_bone = (ct_smooth > self.config["bone_core_threshold"]) & combined_mask
        
        bone_dilated = ndimage.binary_dilation(core_bone, iterations=self.config["dilation_iter"])
        cartilage = (ct_smooth > self.config["cartilage_threshold"]) & bone_dilated & combined_mask
        bone_mask = core_bone | cartilage
        
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
        float_mask = ndimage.gaussian_filter(bone_mask.astype(np.float32), sigma=self.config["mesh_smooth_sigma"])
        bone_grid = pv.wrap(float_mask)
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
            mesh.save(os.path.join(output_dir, f"{case_name}_V1_{mesh_name}.obj"))