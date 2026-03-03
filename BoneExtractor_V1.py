# BoneExtractor_V1.py - V1稳健基线版 (内脏剥离 + 软骨搭桥 + 适度平滑)
import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv
import os
import sys
import tempfile
import shutil

class BoneExtractorV1:
    """V1版本骨骼提取算法 - 稳定、干净、骨骼连贯"""
    
    def __init__(self, config=None):
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.AI_AVAILABLE = True
        except ImportError:
            print("TotalSegmentator未安装，请运行: pip install TotalSegmentator")
            self.AI_AVAILABLE = False
        
        # === V1 稳健解剖学参数 ===
        self.config = {
            "use_high_resolution": True,
            "ai_task": "total",
            
            # 1. 密度与形态学参数 (核心剥离与软骨搭桥)
            "bone_core_threshold": 150,     # 【去除内脏】核心骨架高门槛，彻底屏蔽内脏和肌肉
            "cartilage_threshold": 80,      # 【软骨搭桥】软骨及格线，用于连接断裂的肋骨
            "dilation_iter": 2,             # 极窄搜索光环，仅向外延展2像素寻找软骨
            "gaussian_sigma": 0.5,          # 适度预处理模糊，去除噪点
            "closing_radius": 2,            # 适度闭运算，填补骨头之间的微小裂缝
            "min_component_size": 1500,
            
            # 2. 3D建模与平滑参数 (舒适的视觉平滑度)
            "mesh_smooth_sigma": 0.6,       # 亚像素模糊：融化马赛克台阶
            "smooth_iter": 40,              # 平滑次数：适中，既不粗糙也不像塑料
            "smooth_relax": 0.02,           # 松弛度：轻微收敛，让骨骼表面看起来圆润连贯
            "decimate_ratio": 0.1,
            "bone_color": "#f5f5f5"
        }
        
        if config:
            self.config.update(config)
    
    def run_ai_segmentation(self, ct_data, ct_affine, output_dir):
        """使用TotalSegmentator提取初步区域"""
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
    
    # ==================== 新增：主程序调用的接口 ====================
    def extract_bone(self, ct_data, spacing):
        """
        供主程序调用的骨骼提取接口（符合主程序调用习惯）
        参数：
            ct_data: 3D CT数据
            spacing: 体素间距 (x, y, z)
        返回：
            骨骼掩码 (3D bool数组)
        """
        print(">>> V1 自动运行AI分割并提取骨骼 <<<")
        
        # 1. 根据spacing构造简单的仿射矩阵（用于AI分割）
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # 2. 创建临时目录存放AI分割结果
        temp_dir = tempfile.mkdtemp(prefix="V1_AI_")
        try:
            # 3. 运行AI分割，获取AI结果路径
            ai_output_path = self.run_ai_segmentation(ct_data, affine, temp_dir)
            
            # 4. 调用内部带AI的提取方法
            bone_mask = self._extract_bone_with_ai(ct_data, affine, spacing, ai_output_path)
            
            return bone_mask
        finally:
            # 5. 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ==================== 原extract_bone重命名为内部方法 ====================
    def _extract_bone_with_ai(self, ct_data, ct_affine, spacing, ai_output_path):
        """V1混合算法：高阈值剔内脏 + 光环外扩找软骨（需要AI结果）"""
        print(">>> 开始 V1 稳健版骨骼提取（使用AI结果）<<<")
        t0 = time.time()
        import nibabel as nib
        
        # 1. 加载AI大杂烩
        ai_img = nib.load(ai_output_path)
        ai_data = ai_img.get_fdata()
        
        # 适度高斯平滑CT数据
        ct_smooth = ndimage.gaussian_filter(ct_data, sigma=self.config['gaussian_sigma'])
        
        # ==========================================
        # 🎯 核心逻辑 1：去除内脏 (严格阈值护城河)
        # ==========================================
        # 用 >150 瞬间剔除所有五脏六腑，拿到干净的“绝对骨架”
        core_bone = (ct_smooth > self.config["bone_core_threshold"]) & (ai_data > 0)
        
        # ==========================================
        # 🎯 核心逻辑 2：软骨搭桥 (突破AI盲区)
        # ==========================================
        # 在“绝对骨架”向外膨胀2像素的极窄缝隙里，用 >80 的阈值去抓取软骨
        bone_dilated = ndimage.binary_dilation(core_bone, iterations=self.config["dilation_iter"])
        cartilage = (ct_smooth > self.config["cartilage_threshold"]) & bone_dilated
        
        # 完美拼合：硬骨头 + 关节软骨
        bone_mask = core_bone | cartilage
        
        # 3. 适度形态学闭运算（让骨骼连在一起，修复断裂感）
        bone_closed = morphology.binary_closing(
            bone_mask, 
            morphology.ball(self.config["closing_radius"])
        )
        
        # 4. 连通域过滤（过滤周围飘浮的噪点）
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
            
        # 5. 填充骨髓实心
        final_bone_mask = ndimage.binary_fill_holes(final_mask)
        
        print(f"✅ 骨骼提取完成! 耗时: {time.time()-t0:.2f}秒")
        return final_bone_mask
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """创建连贯、舒适的3D网格"""
        meshes = {}
        
        # 1. 舒适的亚像素模糊 (消除马赛克)
        float_mask = ndimage.gaussian_filter(bone_mask.astype(np.float32), sigma=self.config["mesh_smooth_sigma"])
        bone_grid = pv.wrap(float_mask)
        bone_grid.spacing = spacing
        bone_mesh = bone_grid.contour([0.5])
        
        # 2. 舒适的顶点平滑 (让骨骼看起来不那么尖锐断裂)
        bone_mesh = bone_mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True
        )
        
        bone_mesh = bone_mesh.compute_normals(point_normals=True, auto_orient_normals=True)
        if self.config["decimate_ratio"] > 0:
            bone_mesh = bone_mesh.decimate(self.config["decimate_ratio"])
        
        meshes['bone'] = bone_mesh
        
        # 肿瘤标注生成逻辑
        if label_data is not None and np.any(label_data):
            tumor_float = ndimage.gaussian_filter(label_data.astype(np.float32), sigma=0.8)
            tumor_grid = pv.wrap(tumor_float)
            tumor_grid.spacing = spacing
            tumor_mesh = tumor_grid.contour([0.5]).smooth(n_iter=30, relaxation_factor=0.02)
            tumor_mesh = tumor_mesh.compute_normals(point_normals=True)
            meshes['tumor'] = tumor_mesh
            
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """导出OBJ格式"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        for mesh_name, mesh in meshes.items():
            obj_path = os.path.join(output_dir, f"{case_name}_V1_{mesh_name}.obj")
            try:
                mesh.save(obj_path)
                print(f"✓ 导出: {obj_path} ({mesh.n_cells:,}面)")
            except Exception as e:
                print(f"导出失败: {e}")