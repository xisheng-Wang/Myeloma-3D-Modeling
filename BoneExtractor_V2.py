# BoneExtractor_V2.py - V2纯净结构版 (极致骨架 + 医生标注肿瘤)
import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv
import os
import sys
import tempfile
import shutil

class BoneExtractorV2:
    """V2纯净结构版 - 专注锐利骨架与医生标注肿瘤的物理形态重建"""
    
    def __init__(self, config=None):
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.AI_AVAILABLE = True
        except ImportError:
            print("TotalSegmentator未安装，请运行: pip install TotalSegmentator")
            self.AI_AVAILABLE = False
        
        # === V2 锐利解剖学与高级渲染参数 ===
        self.config = {
            "use_high_resolution": True,
            "ai_task": "total",
            
            # 1. 密度与形态学参数 (保持清爽锐利，只在AI范围内操作)
            "bone_low_threshold": 50,       # 极低阈值保住肋软骨
            "gaussian_sigma": 0.3,                 # 极低预处理模糊，保护CT骨质纹理
            "closing_radius": 1,            # 最小化闭运算
            "min_component_size": 1500,
            
            # 2. 3D建模与平滑参数 (去马赛克而不失真)
            "mesh_smooth_sigma": 0.4,       # 亚像素模糊融化方块感
            "smooth_iter": 20,              # 极低平滑次数，保留真实起伏
            "smooth_relax": 0.01,           # 极低松弛度，防止关节缩水
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
        
        temp_ct_path = os.path.join(output_dir, "temp_ct.nii.gz")
        ai_output_path = os.path.join(output_dir, "ai_segmentation.nii.gz")
        
        if os.path.exists(ai_output_path):
            print("检测到已有的AI分割结果，跳过AI推理")
            return ai_output_path
        
        print(">>> 启动 TotalSegmentator AI 分割 <<<")
        t0 = time.time()
        
        try:
            from totalsegmentator.python_api import totalsegmentator
            temp_img = nib.Nifti1Image(ct_data, ct_affine)
            nib.save(temp_img, temp_ct_path)
            
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
            if os.path.exists(temp_ct_path):
                os.remove(temp_ct_path)
            return ai_output_path
            
        except Exception as e:
            print(f"❌ AI分割失败: {e}")
            if os.path.exists(temp_ct_path):
                os.remove(temp_ct_path)
            raise e
            
    # ==================== 新增：供主程序调用的统一接口 ====================
    def extract_bone(self, ct_data, spacing):
        """
        供主程序调用的骨骼提取接口（完全兼容 main.py）
        """
        print(">>> V2 自动运行AI分割并提取极致纯净骨架 <<<")
        
        # 根据spacing构造简单的仿射矩阵（用于AI分割）
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # 创建临时目录存放AI分割结果
        temp_dir = tempfile.mkdtemp(prefix="V2_AI_")
        try:
            # 运行AI分割，获取AI结果路径
            ai_output_path = self.run_ai_segmentation(ct_data, affine, temp_dir)
            
            # 调用内部的提取逻辑
            bone_mask = self._extract_bone_with_ai(ct_data, affine, spacing, ai_output_path)
            
            return bone_mask
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ==================== 原来的提取逻辑改为内部方法 ====================
    def _extract_bone_with_ai(self, ct_data, ct_affine, spacing, ai_output_path):
        """骨骼提取核心逻辑：绝对边界控制（代码完全没变！）"""
        print(">>> 开始 V2 纯净骨架物理重建 <<<")
        t0 = time.time()
        import nibabel as nib
        
        # 1. 加载AI大杂烩（包含骨头+内脏）
        ai_img = nib.load(ai_output_path)
        ai_data = ai_img.get_fdata()
        ai_region_mask = (ai_data > 0)
        
        # 2. 极轻微高斯平滑CT图像，去噪但不丢细节
        ct_smooth = ndimage.gaussian_filter(ct_data, sigma=self.config['gaussian_sigma'])
        
        # 3. 核心提取：严格限制在AI区域内，用低阈值过滤内脏保留骨骼和软骨
        bone_mask = (ct_smooth > self.config["bone_low_threshold"]) & ai_region_mask
        
        # 4. 微量形态学修复（填补内部极小空隙）
        bone_closed = morphology.binary_closing(
            bone_mask, 
            morphology.ball(self.config["closing_radius"])
        )
        
        # 5. 连通域过滤（过滤周围飘浮的噪点）
        labeled_final, num_final = ndimage.label(bone_closed)
        if num_final > 0:
            component_sizes = ndimage.sum(np.ones_like(bone_closed, dtype=int), labeled_final, range(1, num_final + 1))
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
            
        # 6. 填充实心
        final_bone_mask = ndimage.binary_fill_holes(final_mask)
        
        print(f"✅ 骨骼核心提取完成! 耗时: {time.time()-t0:.2f}秒")
        return final_bone_mask
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """创建网格：仅输出全身骨骼与医生标注肿瘤的物理白模"""
        meshes = {}
        
        print("    -> [Mesh] 正在生成亚像素级锐利骨架...")
        # 1. 骨骼：极其克制的亚像素模糊
        float_mask = ndimage.gaussian_filter(bone_mask.astype(np.float32), sigma=self.config["mesh_smooth_sigma"])
        bone_grid = pv.wrap(float_mask)
        bone_grid.spacing = spacing
        bone_mesh = bone_grid.contour([0.5])
        
        # 极简平滑，防止关节缩水
        bone_mesh = bone_mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True
        )
        
        bone_mesh = bone_mesh.compute_normals(point_normals=True, auto_orient_normals=True)
        if self.config["decimate_ratio"] > 0:
            bone_mesh = bone_mesh.decimate(self.config["decimate_ratio"])
        
        meshes['bone'] = bone_mesh
                
        # ==========================================
        # 🎯 物理形态重建：医生标注的骨髓瘤模型
        # ==========================================
        if label_data is not None and np.any(label_data):
            print("    -> [Mesh] 正在生成医生标注肿瘤的实体网格...")
            tumor_float = ndimage.gaussian_filter(label_data.astype(np.float32), sigma=0.8)
            tumor_grid = pv.wrap(tumor_float)
            tumor_grid.spacing = spacing
            # 肿瘤通常是软组织形态，所以平滑参数给得比骨头稍微大一点，呈现圆润的肉块感
            tumor_mesh = tumor_grid.contour([0.5]).smooth(n_iter=30, relaxation_factor=0.02)
            tumor_mesh = tumor_mesh.compute_normals(point_normals=True)
            meshes['tumor'] = tumor_mesh
            
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """统一导出为OBJ格式"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        for mesh_name, mesh in meshes.items():
            obj_path = os.path.join(output_dir, f"{case_name}_V2_{mesh_name}.obj")
            try:
                mesh.save(obj_path)
                print(f"✓ 导出: {obj_path} ({mesh.n_cells:,}面)")
            except Exception as e:
                print(f"导出失败: {e}")