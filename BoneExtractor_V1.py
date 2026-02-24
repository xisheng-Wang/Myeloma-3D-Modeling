# BoneExtractor_V1.py - 原始版本算法
import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv

class BoneExtractorV1:
    """原始版本骨骼提取算法（old.py版本）"""
    
    def __init__(self, config=None):
        # 默认参数配置
        self.config = {
            "bone_threshold": 130,
            "bed_removal_radius": 0.45,
            "smooth_iter": 50,
            "tumor_opacity": 1.0,
            "bone_opacity": 0.35
        }
        
        if config:
            self.config.update(config)
    
    def extract_bone(self, ct_data, spacing=None):
        """高保真骨骼提取算法"""
        print("开始提取高保真全身骨骼...")
        t0 = time.time()
        
        # 1. 生成圆柱体掩码
        d, h, w = ct_data.shape
        cy, cx = h // 2, w // 2
        y = np.arange(h)
        x = np.arange(w)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        radius = w * self.config["bed_removal_radius"]
        cylinder_mask_slice = dist < radius
        cylinder_mask = np.broadcast_to(cylinder_mask_slice, (d, h, w))
        
        # 2. 阈值分割
        bone_candidate = (ct_data > self.config["bone_threshold"]) & cylinder_mask
        
        # 3. 形态学优化
        bone_clean = morphology.binary_opening(bone_candidate, morphology.ball(1))
        
        # 4. 连通域提取
        labeled, num_features = ndimage.label(bone_clean)
        if num_features > 0:
            slices = ndimage.find_objects(labeled)
            volumes = [np.sum(labeled[s] == (i+1)) for i, s in enumerate(slices)]
            sorted_indices = np.argsort(volumes)[::-1]
            top_n = min(5, len(sorted_indices))
            
            final_bone_mask = np.zeros_like(bone_clean, dtype=bool)
            for i in range(top_n):
                idx = sorted_indices[i]
                if volumes[idx] < volumes[sorted_indices[0]] * 0.01:
                    continue
                final_bone_mask |= (labeled == (idx + 1))
        else:
            final_bone_mask = bone_clean
        
        # 5. 内部填充
        final_bone_mask = ndimage.binary_fill_holes(final_bone_mask)
        
        print(f"骨骼提取完成. 耗时: {time.time()-t0:.2f}s")
        return final_bone_mask
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """创建3D网格"""
        meshes = {}
        
        # 骨骼网格
        bone_grid = pv.wrap(bone_mask.astype(np.uint8))
        bone_grid.spacing = spacing
        bone_mesh = bone_grid.contour([0.5])
        bone_mesh = bone_mesh.smooth(n_iter=self.config["smooth_iter"], relaxation_factor=0.1)
        bone_mesh = bone_mesh.decimate(0.1)
        meshes['bone'] = bone_mesh
        
        # 医生标注网格
        if label_data is not None and np.any(label_data):
            lbl_grid = pv.wrap(label_data.astype(np.uint8))
            lbl_grid.spacing = spacing
            label_mesh = lbl_grid.contour([0.5])
            label_mesh = label_mesh.smooth(n_iter=30)
            meshes['label'] = label_mesh
        
        # PET热点网格
        if pet_data is not None:
            pet_threshold = np.max(pet_data) * 0.4
            pet_mask = (pet_data > pet_threshold)
            if np.any(pet_mask):
                pet_grid = pv.wrap(pet_mask.astype(np.uint8))
                pet_grid.spacing = spacing
                pet_mesh = pet_grid.contour([0.5]).smooth(n_iter=10)
                meshes['pet'] = pet_mesh
        
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """导出OBJ模型"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for mesh_name, mesh in meshes.items():
            file_path = os.path.join(output_dir, f"{case_name}_V1_{mesh_name}.obj")
            mesh.save(file_path)
            print(f"导出: {file_path}")