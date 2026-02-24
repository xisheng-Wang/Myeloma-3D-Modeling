# BoneExtractor_V3.py - Unity优化版本算法
import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv

class BoneExtractorV3:
    """Unity优化版本骨骼提取算法（test09.py版本）- 大创竞赛专用版"""
    
    def __init__(self, config=None):
        # === [针对 Unity 优化的参数] ===
        self.config = {
            "gaussian_sigma": 1.0,
            "bed_removal_radius": 0.48,
            "bone_high_threshold": 220,
            "bone_low_threshold": 80,
            "morph_kernel_size": 3,
            "smooth_iter": 150,
            "pass_band": 0.05,
            "tumor_opacity": 1.0,
            "bone_opacity": 1.0
        }
        
        if config:
            self.config.update(config)
    
    def extract_bone(self, ct_data, spacing=None):
        """[大创算法核心] 修复断裂、去除床板、保留手臂"""
        print(">>> 开始执行 Unity 级骨骼提取 <<<")
        t0 = time.time()
        
        # 1. 强力高斯平滑
        ct_smooth = ndimage.gaussian_filter(ct_data, sigma=self.config['gaussian_sigma'])
        
        # 2. 宽容 ROI 裁剪
        d, h, w = ct_smooth.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask_roi = ((x - cx)**2 + (y - cy)**2) < (w * self.config["bed_removal_radius"])**2
        mask_roi = mask_roi[np.newaxis, :, :]
        
        # 3. 迟滞阈值
        mask_high = (ct_smooth > self.config["bone_high_threshold"]) & mask_roi
        mask_low = (ct_smooth > self.config["bone_low_threshold"]) & mask_roi
        
        # 4. 连通性重建
        labeled_low, _ = ndimage.label(mask_low)
        valid_labels = np.unique(labeled_low[mask_high])
        valid_labels = valid_labels[valid_labels != 0]
        bone_mask = np.isin(labeled_low, valid_labels) if len(valid_labels) > 0 else mask_high
        
        # 5. 强力形态学闭运算
        struct_elem = morphology.ball(self.config["morph_kernel_size"])
        bone_closed = morphology.binary_closing(bone_mask, struct_elem)
        
        # 6. 智能开运算
        bone_opened = morphology.binary_opening(bone_closed, morphology.ball(1))
        
        # 7. 最大连通域提取
        labeled_final, num_final = ndimage.label(bone_opened)
        if num_final > 0:
            slices = ndimage.find_objects(labeled_final)
            
            # 寻找体积最大的块
            volumes = []
            for i, s in enumerate(slices):
                if s is not None:
                    vol = np.sum(labeled_final[s] == (i+1))
                    volumes.append(vol)
            
            sorted_indices = np.argsort(volumes)[::-1]
            
            # 保留前 3 大块，且体积必须大于最大块的 5%
            final_bone_mask = np.zeros_like(bone_opened, dtype=bool)
            max_vol = volumes[sorted_indices[0]] if volumes else 0
            
            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i] + 1
                vol = volumes[sorted_indices[i]]
                if vol > max_vol * 0.05:
                    final_bone_mask |= (labeled_final == idx)
        else:
            final_bone_mask = bone_opened
            
        # 8. 内部填洞
        final_bone_mask = ndimage.binary_fill_holes(final_bone_mask)
        
        print(f"提取耗时: {time.time()-t0:.2f}s")
        return final_bone_mask
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """生成 Unity 友好的 Mesh"""
        meshes = {}
        
        # 骨骼网格
        grid = pv.wrap(bone_mask.astype(np.uint8))
        grid.spacing = spacing
        mesh = grid.contour([0.5])
        
        # 应用迭代微松弛平滑
        mesh = mesh.smooth(
            n_iter=100,
            relaxation_factor=0.05,
            boundary_smoothing=True
        )
        
        # Unity关键步骤：计算顶点法线
        mesh = mesh.compute_normals(
            cell_normals=False, 
            point_normals=True, 
            split_vertices=True,
            auto_orient_normals=True
        )
        
        # 简化网格
        meshes['bone'] = mesh.decimate(0.1)
        
        # 肿瘤网格
        if label_data is not None and np.any(label_data):
            l_grid = pv.wrap(label_data.astype(np.uint8))
            l_grid.spacing = spacing
            l_mesh = l_grid.contour([0.5]).smooth(n_iter=50, relaxation_factor=0.1)
            l_mesh = l_mesh.compute_normals(point_normals=True, auto_orient_normals=True)
            meshes['label'] = l_mesh
        
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """导出为 Unity 可用的 OBJ 文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, mesh in meshes.items():
            file_path = os.path.join(output_dir, f"{case_name}_V3_{name}.obj")
            
            # 确保法线存在
            if 'Normals' not in mesh.point_data:
                mesh = mesh.compute_normals(point_normals=True)
            
            mesh.save(file_path)
            print(f"导出: {file_path}")