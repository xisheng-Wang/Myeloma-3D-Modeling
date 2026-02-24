# BoneExtractor_AI.py - AI增强版本算法
import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv
import os
import sys

class BoneExtractorAI:
    """AI增强版本骨骼提取算法"""
    
    def __init__(self, config=None):
        # 检查TotalSegmentator是否可用
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.AI_AVAILABLE = True
        except ImportError:
            print("TotalSegmentator未安装，请运行: pip install TotalSegmentator")
            self.AI_AVAILABLE = False
        
        # === AI优化参数 ===
        self.config = {
            "use_high_resolution": True,
            "ai_task": "total",
            "bone_high_threshold": 250,
            "bone_low_threshold": 90,
            "gaussian_sigma": 0.5,
            "min_component_size": 1500,
            "closing_radius": 1,
            "smooth_iter": 80,
            "smooth_relax": 0.04,
            "decimate_ratio": 0.1,
            "bone_color": "#f5f5f5"
        }
        
        if config:
            self.config.update(config)
    
    def run_ai_segmentation(self, ct_data, ct_affine, output_dir):
        """使用TotalSegmentator分割人体区域"""
        if not self.AI_AVAILABLE:
            raise ImportError("TotalSegmentator未安装")
        
        import nibabel as nib
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存临时CT文件供AI使用
        temp_ct_path = os.path.join(output_dir, "temp_ct.nii.gz")
        ai_output_path = os.path.join(output_dir, "ai_segmentation.nii.gz")
        
        # 检查是否已有AI结果
        if os.path.exists(ai_output_path):
            print("检测到已有的AI分割结果，跳过AI推理")
            return ai_output_path
        
        print(">>> 启动 TotalSegmentator AI 分割 <<<")
        t0 = time.time()
        
        try:
            from totalsegmentator.python_api import totalsegmentator
            
            # 保存临时CT
            temp_img = nib.Nifti1Image(ct_data, ct_affine)
            nib.save(temp_img, temp_ct_path)
            
            # 调用TotalSegmentator
            totalsegmentator(
                input=temp_ct_path,
                output=ai_output_path,
                fast=not self.config['use_high_resolution'],
                ml=True,
                task=self.config['ai_task'],
                preview=False,
                force_split=False
            )
            
            print(f"✅ AI分割完成! 耗时: {time.time()-t0:.2f}秒")
            
            # 清理临时文件
            if os.path.exists(temp_ct_path):
                os.remove(temp_ct_path)
            
            return ai_output_path
            
        except Exception as e:
            print(f"❌ AI分割失败: {e}")
            if os.path.exists(temp_ct_path):
                os.remove(temp_ct_path)
            raise e
    
    def extract_bone(self, ct_data, ct_affine, spacing, ai_output_path):
        """混合算法：AI区域 + 精确阈值分割"""
        print(">>> 开始AI增强骨骼提取 <<<")
        t0 = time.time()
        
        import nibabel as nib
        
        # 1. 加载AI分割结果
        ai_img = nib.load(ai_output_path)
        ai_data = ai_img.get_fdata()
        ai_region_mask = (ai_data > 0)
        
        # 2. 轻度高斯平滑
        ct_smooth = ndimage.gaussian_filter(ct_data, sigma=self.config['gaussian_sigma'])
        
        # 3. 迟滞阈值分割（限制在AI区域内）
        mask_high = (ct_smooth > self.config["bone_high_threshold"]) & ai_region_mask
        mask_low = (ct_smooth > self.config["bone_low_threshold"]) & ai_region_mask
        
        # 4. 连通性重建
        labeled_low, num_features = ndimage.label(mask_low)
        valid_labels = np.unique(labeled_low[mask_high])
        valid_labels = valid_labels[valid_labels != 0]
        
        if len(valid_labels) == 0:
            bone_mask = mask_high
        else:
            bone_mask = np.isin(labeled_low, valid_labels)
        
        # 5. 形态学修复
        bone_closed = morphology.binary_closing(
            bone_mask, 
            morphology.ball(self.config["closing_radius"])
        )
        
        # 6. 连通域过滤
        labeled_final, num_final = ndimage.label(bone_closed)
        
        if num_final > 0:
            # 计算每个连通域的大小
            component_sizes = ndimage.sum(
                np.ones_like(bone_closed, dtype=int),
                labeled_final,
                range(1, num_final + 1)
            )
            
            # 按大小排序，保留大的连通域
            sorted_indices = np.argsort(component_sizes)[::-1] + 1
            final_mask = np.zeros_like(bone_closed, dtype=bool)
            
            keep_count = 0
            for idx in sorted_indices:
                if component_sizes[idx-1] < self.config["min_component_size"]:
                    continue
                
                final_mask |= (labeled_final == idx)
                keep_count += 1
                
                if keep_count >= 5:
                    break
        else:
            final_mask = bone_closed
        
        # 7. 填充内部空洞
        final_bone_mask = ndimage.binary_fill_holes(final_mask)
        
        print(f"✅ 骨骼提取完成! 耗时: {time.time()-t0:.2f}秒")
        return final_bone_mask
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """创建高质量的3D网格"""
        meshes = {}
        
        # 1. 骨骼网格
        bone_grid = pv.wrap(bone_mask.astype(np.uint8))
        bone_grid.spacing = spacing
        bone_mesh = bone_grid.contour([0.5])
        
        # 微松弛平滑
        bone_mesh = bone_mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True
        )
        
        # 计算法线（Unity必需）
        bone_mesh = bone_mesh.compute_normals(
            point_normals=True,
            auto_orient_normals=True
        )
        
        # 简化网格
        if self.config["decimate_ratio"] > 0:
            bone_mesh = bone_mesh.decimate(self.config["decimate_ratio"])
        
        meshes['bone'] = bone_mesh
        
        # 2. PET热点网格
        if pet_data is not None:
            pet_max = np.max(pet_data)
            pet_threshold = pet_max * 0.4
            pet_mask = (pet_data > pet_threshold)
            
            if np.any(pet_mask):
                pet_grid = pv.wrap(pet_mask.astype(np.uint8))
                pet_grid.spacing = spacing
                pet_mesh = pet_grid.contour([0.5])
                pet_mesh = pet_mesh.smooth(n_iter=15, relaxation_factor=0.1)
                pet_mesh = pet_mesh.compute_normals(point_normals=True)
                meshes['pet'] = pet_mesh
        
        # 3. 医生标注网格
        if label_data is not None and np.any(label_data):
            tumor_grid = pv.wrap(label_data.astype(np.uint8))
            tumor_grid.spacing = spacing
            tumor_mesh = tumor_grid.contour([0.5])
            tumor_mesh = tumor_mesh.smooth(n_iter=5, relaxation_factor=0.01)
            tumor_mesh = tumor_mesh.compute_normals(point_normals=True)
            meshes['tumor'] = tumor_mesh
        
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """导出为Unity可用的OBJ格式"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for mesh_name, mesh in meshes.items():
            obj_path = os.path.join(output_dir, f"{case_name}_AI_{mesh_name}.obj")
            
            try:
                mesh.save(obj_path)
                print(f"✓ 导出: {obj_path} ({mesh.n_cells:,}面)")
            except Exception as e:
                print(f"导出失败: {e}")