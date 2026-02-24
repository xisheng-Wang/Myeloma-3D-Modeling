import numpy as np
from scipy import ndimage, signal
from skimage import morphology, measure, filters
import time
import pyvista as pv
import os
import sys
import nibabel as nib
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class BoneExtractorV2:
    """
    [V2 旗舰混合版] AI 语义分割 + 密度智能缝合 + 高级骨髓瘤分析
    
    设计目标：解决 AI"断裂" 和 V3"粘连" 的矛盾
    核心逻辑：
    1. 使用 AI 提取核心骨架（保证无床板、大结构准确）。
    2. 使用 "搜索扩张" 技术，在 AI 骨架周围寻找被漏掉的低密度骨（修复断裂）。
    3. 高级骨髓瘤分析：SUV自适应阈值、置信度评分、热力图
    """
    
    def __init__(self, config=None):
        # 检查 TotalSegmentator
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.AI_AVAILABLE = True
        except ImportError:
            print("【严重警告】TotalSegmentator 未安装！无法运行 AI 混合算法。")
            self.AI_AVAILABLE = False

        # === 混合动力参数 ===
        self.config = {
            # 1. AI 参数
            "use_high_res_ai": True,    # 必须用高分辨率，否则边缘会有马赛克
            
            # 2. 智能缝合参数 (Smart Stitching)
            "stitch_threshold": 150,     # 补全阈值 (比 V3 的 220 低，为了抓取断裂的薄骨)
            "dilation_iter": 2,           # 搜索范围：在 AI 骨架周围 2 个体素范围内搜索漏网之鱼
            
            # 3. 高级骨骼参数
            "cortical_bone_threshold": 250,    # 皮质骨阈值
            "trabecular_bone_threshold": 150,  # 松质骨阈值
            "bone_low_threshold": 90,           # 骨骼低阈值
            
            # 4. 骨髓瘤专用参数
            "tumor_suv_min": 2.5,               # 最小SUV值
            "tumor_volume_min": 0.5,             # 最小肿瘤体积(cm³)
            "tumor_confidence_threshold": 0.6,   # 肿瘤置信度阈值
            "tumor_merge_distance": 5,            # 肿瘤合并距离(mm)
            
            # 5. 形态学参数
            "morph_radius": 1,                    # 形态学半径
            "min_component_size": 500,             # 最小连通域大小(体素)
            
            # 6. Unity 美化
            "smooth_iter": 100,                    # 平滑迭代
            "smooth_relax": 0.05,                   # 低松弛，保体积
            "decimate_ratio": 0.15,                  # 减面比例
            "feature_angle": 60,                     # 特征保留角度
            
            # 7. 可视化参数
            "bone_color": "#E8D8C0",                 # 骨骼颜色
            "tumor_colormap": "hot",                  # 肿瘤热力图
            "tumor_opacity": 0.9,                     # 肿瘤透明度
        }
        
        if config:
            self.config.update(config)
            
        # 初始化SUV转换因子
        self.suv_factor = 1.0
    
    def _run_internal_ai(self, ct_data, output_dir=None):
        """运行 AI 获取纯净骨架"""
        if not self.AI_AVAILABLE:
            raise ImportError("需要 TotalSegmentator")

        from totalsegmentator.python_api import totalsegmentator
        
        if output_dir is None:
            import tempfile
            output_dir = tempfile.gettempdir()
            
        temp_ct_path = os.path.join(output_dir, "_temp_hybrid_ct.nii.gz")
        ai_mask_path = os.path.join(output_dir, "_temp_hybrid_mask.nii.gz")
        
        try:
            # 构造临时 CT
            affine = np.eye(4) 
            nib.save(nib.Nifti1Image(ct_data, affine), temp_ct_path)
            
            # 运行 AI
            is_fast = not self.config['use_high_res_ai']
            print(f"    -> [AI引擎] 正在生成核心骨架 (HighRes=True)...")
            
            totalsegmentator(
                temp_ct_path, 
                ai_mask_path, 
                fast=is_fast, 
                ml=True, 
                task="total",
                quiet=False
            )
            
            ai_img = nib.load(ai_mask_path)
            ai_mask = ai_img.get_fdata() > 0
            
            return ai_mask
            
        except Exception as e:
            print(f"    -> [AI Error]: {e}")
            raise e
        finally:
            if os.path.exists(temp_ct_path): os.remove(temp_ct_path)
            if os.path.exists(ai_mask_path): os.remove(ai_mask_path)

    def extract_bone(self, ct_data, spacing=None):
        """
        [混合算法核心] AI + 密度补全 + 多尺度骨骼提取
        """
        print(">>> [V2 旗舰混合版] 启动智能修复提取 <<<")
        t0 = time.time()
        
        # 1. 预处理：各向异性滤波保边平滑
        ct_smooth = self._anisotropic_diffusion(ct_data, iterations=3)
        
        # 2. 获取 AI 核心骨架 (绝对纯净，无床板)
        ai_mask = self._run_internal_ai(ct_data, output_dir=os.getcwd())
        
        # 确保尺寸一致
        if ai_mask.shape != ct_data.shape:
            print("    -> [警告] 尺寸对齐修正...")
            temp_mask = np.zeros_like(ct_data, dtype=bool)
            min_d = min(ai_mask.shape[0], ct_data.shape[0])
            min_h = min(ai_mask.shape[1], ct_data.shape[1])
            min_w = min(ai_mask.shape[2], ct_data.shape[2])
            temp_mask[:min_d, :min_h, :min_w] = ai_mask[:min_d, :min_h, :min_w]
            ai_mask = temp_mask
        
        # 3. [关键步骤] 智能缝合 (Smart Stitching)
        print(f"    -> [缝合] 正在修复 AI 断裂区域 (Threshold > {self.config['stitch_threshold']})...")
        
        # a. 扩张 AI 区域 (建立搜索区)
        search_zone = ndimage.binary_dilation(ai_mask, iterations=self.config['dilation_iter'])
        
        # b. 在搜索区内，找回被遗漏的高密度像素
        missed_bone = search_zone & (ct_smooth > self.config['stitch_threshold'])
        
        # c. 合并：原始 AI + 找回的骨头
        hybrid_mask = ai_mask | missed_bone
        
        # 4. 微创形态学修复
        print(f"    -> [微调] 形态学闭运算 (Radius={self.config['morph_radius']})...")
        final_mask = morphology.binary_closing(
            hybrid_mask, 
            morphology.ball(self.config['morph_radius'])
        )
        
        # 5. 连通域分析，去除小噪声
        final_mask = self._component_analysis(final_mask)
        
        # 6. 内部填实
        print("    -> [加固] 填充内部空洞...")
        final_mask = ndimage.binary_fill_holes(final_mask)
        
        # 7. 分离皮质骨和松质骨（用于高级可视化）
        cortical_mask = (ct_smooth > self.config["cortical_bone_threshold"]) & final_mask
        trabecular_mask = final_mask & ~cortical_mask & \
                         (ct_smooth > self.config["trabecular_bone_threshold"])
        
        print(f"✅ V2 混合提取完成. 耗时: {time.time()-t0:.2f}s")
        print(f"   皮质骨体素: {np.sum(cortical_mask):,}")
        print(f"   松质骨体素: {np.sum(trabecular_mask):,}")
        
        # 返回完整掩码，同时保存子掩码供后续使用
        self.bone_masks = {
            'full': final_mask,
            'cortical': cortical_mask,
            'trabecular': trabecular_mask
        }
        
        return final_mask
    
    def extract_tumors_advanced(self, pet_data, ct_data, spacing, bone_mask=None):
        """
        高级骨髓瘤分割
        
        Args:
            pet_data: PET数据(SUV值)
            ct_data: CT数据
            spacing: 体素间距
            bone_mask: 骨骼掩码(可选)
            
        Returns:
            tumor_info: 包含肿瘤掩码和特征字典
        """
        print(">>> [V2] 开始高级骨髓瘤分析 <<<")
        t0 = time.time()
        
        if pet_data is None:
            print("PET数据为空，跳过肿瘤分析")
            return None
            
        # 使用已有的骨骼掩码
        if bone_mask is None and hasattr(self, 'bone_masks'):
            bone_mask = self.bone_masks['full']
        
        # 1. 计算体素体积(cm³)
        voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0  # mm³ -> cm³
        
        # 2. 计算SUV统计
        suv_max = np.max(pet_data)
        suv_mean = np.mean(pet_data[pet_data > 0])
        suv_std = np.std(pet_data[pet_data > 0])
        
        print(f"   SUV统计: max={suv_max:.2f}, mean={suv_mean:.2f}, std={suv_std:.2f}")
        
        # 3. 自适应阈值分割
        pet_flat = pet_data[pet_data > pet_data.max() * 0.1]
        if len(pet_flat) > 0:
            threshold = filters.threshold_otsu(pet_flat)
            threshold = max(threshold, self.config["tumor_suv_min"])
        else:
            threshold = self.config["tumor_suv_min"]
        
        print(f"   Otsu自适应阈值: {threshold:.2f}")
        
        # 4. 初步肿瘤掩码
        tumor_candidate = pet_data > threshold
        
        # 5. 空间约束
        if bone_mask is not None:
            bone_dilated = ndimage.binary_dilation(bone_mask, iterations=5)
            tumor_in_bone = tumor_candidate & bone_dilated
        else:
            tumor_in_bone = tumor_candidate
        
        # 6. 连通域分析和体积过滤
        labeled_tumor, num_tumors = ndimage.label(tumor_in_bone)
        
        tumor_masks = {}
        tumor_features = []
        
        for i in range(1, num_tumors + 1):
            tumor_mask = labeled_tumor == i
            tumor_voxels = np.sum(tumor_mask)
            tumor_vol = tumor_voxels * voxel_volume
            
            if tumor_vol < self.config["tumor_volume_min"]:
                continue
                
            tumor_suv_values = pet_data[tumor_mask]
            features = {
                'volume': tumor_vol,
                'suv_max': np.max(tumor_suv_values),
                'suv_mean': np.mean(tumor_suv_values),
                'suv_std': np.std(tumor_suv_values),
                'suv_peak': np.percentile(tumor_suv_values, 95),
                'voxel_count': tumor_voxels,
                'center': ndimage.center_of_mass(tumor_mask)
            }
            
            if bone_mask is not None:
                features['in_bone_ratio'] = np.sum(tumor_mask & bone_mask) / tumor_voxels
            
            confidence = self._calculate_tumor_confidence(features, suv_max)
            features['confidence'] = confidence
            
            if confidence > self.config["tumor_confidence_threshold"]:
                tumor_masks[f'tumor_{i}'] = tumor_mask
                tumor_features.append(features)
        
        print(f"✅ 肿瘤分析完成: 发现 {len(tumor_masks)} 个显著病灶")
        print(f"   耗时: {time.time()-t0:.2f}s")
        
        self.tumor_info = {
            'masks': tumor_masks,
            'features': tumor_features,
            'stats': {
                'suv_max': suv_max,
                'suv_mean': suv_mean,
                'threshold': threshold
            }
        }
        
        return self.tumor_info
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """生成 Unity 友好网格（增强版）"""
        meshes = {}
        
        print("    -> [Mesh] 生成高保真网格...")
        
        # 1. 基础骨骼网格
        grid = pv.wrap(bone_mask.astype(np.uint8))
        grid.spacing = spacing
        mesh = grid.contour([0.5])
        
        # 智能平滑
        print(f"    -> [Unity] 智能平滑 (Iter={self.config['smooth_iter']})...")
        mesh = self._smart_smoothing(mesh)
        
        # 计算法线
        print("    -> [Unity] 烘焙法线...")
        mesh = mesh.compute_normals(
            point_normals=True, 
            split_vertices=True, 
            auto_orient_normals=True,
            feature_angle=self.config["feature_angle"]
        )
        
        # 减面
        if self.config["decimate_ratio"] > 0:
            mesh = mesh.decimate_pro(
                self.config["decimate_ratio"],
                feature_angle=self.config["feature_angle"],
                preserve_topology=True
            )
        
        # 添加颜色信息
        mesh.cell_data['rgb'] = self._hex_to_rgb(self.config["bone_color"])
        meshes['bone'] = mesh
        
        # 2. 皮质骨高亮（如果有）
        if hasattr(self, 'bone_masks') and np.sum(self.bone_masks['cortical']) > 0:
            cortical_grid = pv.wrap(self.bone_masks['cortical'].astype(np.uint8))
            cortical_grid.spacing = spacing
            cortical_mesh = cortical_grid.contour([0.5])
            cortical_mesh = cortical_mesh.smooth(n_iter=30)
            cortical_mesh = cortical_mesh.compute_normals(point_normals=True)
            meshes['cortical'] = cortical_mesh
        
        # 3. 肿瘤网格（带SUV热力图）
        if pet_data is not None:
            # 运行高级肿瘤分析（如果还没运行）
            if not hasattr(self, 'tumor_info'):
                self.extract_tumors_advanced(pet_data, None, spacing, bone_mask)
            
            if hasattr(self, 'tumor_info') and self.tumor_info and 'masks' in self.tumor_info:
                tumor_meshes = []
                
                for tumor_id, tumor_mask in self.tumor_info['masks'].items():
                    tumor_grid = pv.wrap(tumor_mask.astype(np.uint8))
                    tumor_grid.spacing = spacing
                    tumor_mesh = tumor_grid.contour([0.5])
                    tumor_mesh = tumor_mesh.smooth(n_iter=20, relaxation_factor=0.1)
                    tumor_mesh = tumor_mesh.compute_normals(point_normals=True)
                    
                    # SUV热力图颜色
                    tumor_idx = int(tumor_id.split('_')[1]) - 1
                    if tumor_idx < len(self.tumor_info['features']):
                        features = self.tumor_info['features'][tumor_idx]
                        color = self._suv_to_color(
                            features['suv_mean'],
                            self.tumor_info['stats']['suv_max']
                        )
                        tumor_mesh.cell_data['rgb'] = color
                    
                    tumor_meshes.append(tumor_mesh)
                
                if tumor_meshes:
                    combined_tumor = tumor_meshes[0]
                    for tm in tumor_meshes[1:]:
                        combined_tumor = combined_tumor.merge(tm)
                    meshes['tumor'] = combined_tumor
        
        # 4. 医生标注（保持原有逻辑）
        if label_data is not None and np.any(label_data):
            lbl_grid = pv.wrap(label_data.astype(np.uint8))
            lbl_grid.spacing = spacing
            lbl_mesh = lbl_grid.contour([0.5]).smooth(n_iter=20)
            lbl_mesh = lbl_mesh.compute_normals(point_normals=True)
            meshes['label'] = lbl_mesh
        
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """导出 OBJ（增强版，支持GLB格式）"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for mesh_name, mesh in meshes.items():
            if mesh_name == 'tumor':
                # 肿瘤使用GLB格式（支持颜色）
                file_path = os.path.join(output_dir, f"{case_name}_V2_{mesh_name}.glb")
            else:
                file_path = os.path.join(output_dir, f"{case_name}_V2_{mesh_name}.obj")
            
            mesh.save(file_path)
            print(f"导出: {file_path}")
            
            # 为骨骼导出材质文件
            if mesh_name == 'bone':
                self._export_material(output_dir, case_name)
    
    # ========== 私有辅助方法 ==========
    
    def _anisotropic_diffusion(self, data, iterations=3, kappa=50):
        """各向异性扩散滤波(保边平滑)"""
        result = data.copy()
        
        for _ in range(iterations):
            grad_x = np.diff(result, axis=2, append=0)
            grad_y = np.diff(result, axis=1, append=0)
            grad_z = np.diff(result, axis=0, append=0)
            
            c_x = np.exp(-(grad_x/kappa)**2)
            c_y = np.exp(-(grad_y/kappa)**2)
            c_z = np.exp(-(grad_z/kappa)**2)
            
            flux_x = c_x * grad_x
            flux_y = c_y * grad_y
            flux_z = c_z * grad_z
            
            result = result + 0.125 * (
                np.diff(flux_x, axis=2, prepend=0) +
                np.diff(flux_y, axis=1, prepend=0) +
                np.diff(flux_z, axis=0, prepend=0)
            )
        
        return result
    
    def _component_analysis(self, mask):
        """连通域分析，保留主要骨骼"""
        labeled, num = ndimage.label(mask)
        
        if num > 0:
            sizes = ndimage.sum(np.ones_like(mask), labeled, range(1, num + 1))
            sorted_idx = np.argsort(sizes)[::-1]
            
            max_size = sizes[sorted_idx[0]]
            keep_labels = []
            
            for idx in sorted_idx:
                if sizes[idx] >= max_size * 0.05:
                    keep_labels.append(idx + 1)
            
            result = np.isin(labeled, keep_labels)
        else:
            result = mask
            
        return result
    
    def _smart_smoothing(self, mesh):
        """智能平滑(保特征)"""
        # Taubin平滑（无收缩）
        mesh_smooth = mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True
        )
        return mesh_smooth
    
    def _calculate_tumor_confidence(self, features, global_suv_max):
        """计算肿瘤置信度分数"""
        score = 0.0
        
        suv_ratio = features['suv_peak'] / global_suv_max
        score += suv_ratio * 0.5
        
        vol_score = min(features['volume'] / 10.0, 1.0)
        score += vol_score * 0.3
        
        if 'in_bone_ratio' in features:
            score += features['in_bone_ratio'] * 0.2
        
        return min(score, 1.0)
    
    def _suv_to_color(self, suv_value, suv_max):
        """SUV值映射到颜色(RGB)"""
        try:
            import matplotlib.cm as cm
            norm_value = suv_value / suv_max
            rgba = cm.hot(norm_value)
            return (rgba[:3] * 255).astype(np.uint8)
        except:
            intensity = int(255 * (suv_value / suv_max))
            return np.array([255, intensity, intensity])
    
    def _hex_to_rgb(self, hex_color):
        """十六进制颜色转RGB"""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])
    
    def _export_material(self, output_dir, case_name):
        """导出Unity材质文件"""
        mtl_path = os.path.join(output_dir, f"{case_name}_V2_bone.mat")
        
        mtl_content = f"""Shader "Universal Render Pipeline/Lit"
Properties
{{
    _BaseColor("Color", color) = (0.91,0.85,0.75,0.8)
    _SurfaceType("Surface Type", float) = 1
    _BlendMode("Blend Mode", float) = 0
}}
"""
        with open(mtl_path, 'w') as f:
            f.write(mtl_content)