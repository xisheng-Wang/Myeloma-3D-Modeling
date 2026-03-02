# BoneExtractor_V2.py - V2旗舰混合版（极致图形学建模优化版）
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
    [V2 旗舰混合版] AI 语义分割 + 密度智能缝合 + 极致图形学优化
    
    设计目标：解决 AI"断裂" 与传统算法"粘连"的矛盾，并输出顶级渲染质量的 3D 模型。
    核心逻辑：
    1. AI 获取无干扰绝对纯净骨架。
    2. 智能扩张缝合找回骨髓瘤低密度病灶区。
    3. 亚像素平滑去阶梯锯齿 + UV 热力图隐写映射。
    """
    
    def __init__(self, config=None):
        try:
            from totalsegmentator.python_api import totalsegmentator
            self.AI_AVAILABLE = True
        except ImportError:
            print("【严重警告】TotalSegmentator 未安装！无法运行 AI 混合算法。")
            self.AI_AVAILABLE = False

        self.config = {
            "use_high_res_ai": True,                # 坚持最高质量！开启 1.5mm 高清 AI 推理
            "stitch_threshold": 150,     
            "dilation_iter": 2,           
            "cortical_bone_threshold": 250,    
            "trabecular_bone_threshold": 150,  
            "bone_low_threshold": 90,           
            "tumor_suv_min": 2.5,               
            "tumor_volume_min": 0.5,             
            "tumor_confidence_threshold": 0.6,   
            "tumor_merge_distance": 5,            
            "morph_radius": 1,                    
            "min_component_size": 500,             
            "smooth_iter": 150,                     # 提升为高频迭代
            "smooth_relax": 0.015,                  # 极低松弛度，保体积
            "decimate_ratio": 0.1,                  # 拓扑保护减面比例
            "bone_color": "#f5f5f5",                
        }
        
        if config:
            self.config.update(config)
            
        self.suv_factor = 1.0
    
    def _run_internal_ai(self, ct_data, output_dir=None):
        """运行 AI 获取纯净核心骨架"""
        if not self.AI_AVAILABLE:
            raise ImportError("需要 TotalSegmentator")

        from totalsegmentator.python_api import totalsegmentator
        
        if output_dir is None:
            import tempfile
            output_dir = tempfile.gettempdir()
            
        temp_ct_path = os.path.join(output_dir, "_temp_hybrid_ct.nii.gz")
        ai_mask_path = os.path.join(output_dir, "_temp_hybrid_mask.nii.gz")
        
        try:
            affine = np.eye(4) 
            nib.save(nib.Nifti1Image(ct_data, affine), temp_ct_path)
            
            is_fast = not self.config['use_high_res_ai']
            print(f"    -> [AI引擎] 正在生成核心骨架 (HighRes={not is_fast})...")
            
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
        """[混合算法核心] AI 提取 + 密度智能缝合补全"""
        print(">>> [V2 旗舰混合版] 启动智能修复提取 <<<")
        t0 = time.time()
        
        # 1. 各向异性滤波保边平滑
        ct_smooth = self._anisotropic_diffusion(ct_data, iterations=3)
        
        # 2. 运行 AI 拿底图
        print("    -> [AI] 运行语义分割获取核心骨架...")
        ai_mask = self._run_internal_ai(ct_data, output_dir=os.getcwd())
        
        if ai_mask.shape != ct_data.shape:
            print("    -> [警告] 尺寸对齐修正...")
            temp_mask = np.zeros_like(ct_data, dtype=bool)
            min_d = min(ai_mask.shape[0], ct_data.shape[0])
            min_h = min(ai_mask.shape[1], ct_data.shape[1])
            min_w = min(ai_mask.shape[2], ct_data.shape[2])
            temp_mask[:min_d, :min_h, :min_w] = ai_mask[:min_d, :min_h, :min_w]
            ai_mask = temp_mask
        
        # 3. 智能缝合找回低密度骨骼
        print(f"    -> [缝合] 正在修复 AI 断裂区域 (Threshold > {self.config['stitch_threshold']})...")
        search_zone = ndimage.binary_dilation(ai_mask, iterations=self.config['dilation_iter'])
        missed_bone = search_zone & (ct_smooth > self.config['stitch_threshold'])
        hybrid_mask = ai_mask | missed_bone
        
        # 4. 微创形态学修复与连通域过滤
        print(f"    -> [微调] 形态学闭运算 (Radius={self.config['morph_radius']})...")
        final_mask = morphology.binary_closing(
            hybrid_mask, 
            morphology.ball(self.config['morph_radius'])
        )
        final_mask = self._component_analysis(final_mask)
        
        print("    -> [加固] 填充内部空洞...")
        final_mask = ndimage.binary_fill_holes(final_mask)
        
        cortical_mask = (ct_smooth > self.config["cortical_bone_threshold"]) & final_mask
        trabecular_mask = final_mask & ~cortical_mask & (ct_smooth > self.config["trabecular_bone_threshold"])
        
        print(f"✅ V2 混合提取完成. 耗时: {time.time()-t0:.2f}s")
        print(f"   皮质骨体素: {np.sum(cortical_mask):,}")
        print(f"   松质骨体素: {np.sum(trabecular_mask):,}")
        
        self.bone_masks = {
            'full': final_mask,
            'cortical': cortical_mask,
            'trabecular': trabecular_mask
        }
        
        return final_mask
    
    def extract_tumors_advanced(self, pet_data, ct_data=None, spacing=None, bone_mask=None):
        """高级肿瘤分割提取与特征计算"""
        print(">>> [V2] 开始高级骨髓瘤分析 <<<")
        t0 = time.time()
        
        if pet_data is None:
            return None
            
        if bone_mask is None and hasattr(self, 'bone_masks'):
            bone_mask = self.bone_masks['full']
        
        voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0 if spacing else 1.0
        
        suv_max = np.max(pet_data)
        suv_mean = np.mean(pet_data[pet_data > 0])
        
        pet_flat = pet_data[pet_data > pet_data.max() * 0.1]
        if len(pet_flat) > 0:
            threshold = filters.threshold_otsu(pet_flat)
            threshold = max(threshold, self.config["tumor_suv_min"])
        else:
            threshold = self.config["tumor_suv_min"]
        
        tumor_candidate = pet_data > threshold
        
        if bone_mask is not None:
            bone_dilated = ndimage.binary_dilation(bone_mask, iterations=5)
            tumor_in_bone = tumor_candidate & bone_dilated
        else:
            tumor_in_bone = tumor_candidate
        
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
        
        self.tumor_info = {
            'masks': tumor_masks,
            'features': tumor_features,
            'stats': {'suv_max': suv_max, 'suv_mean': suv_mean, 'threshold': threshold}
        }
        return self.tumor_info
    
    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """生成 Unity 友好网格 (极致建模优化版 + 底层字典 UV 隐写)"""
        import pyvista as pv
        from scipy import ndimage
        import numpy as np
        meshes = {}
        
        print("    -> [Mesh] 正在启动亚像素级高保真网格重建...")
        
        # 1. 骨骼网格：亚像素平滑 (消除阶梯锯齿)
        print("    -> [Mesh] 计算亚像素表面场...")
        float_mask = ndimage.gaussian_filter(bone_mask.astype(np.float32), sigma=0.6)
        grid = pv.wrap(float_mask)
        grid.spacing = spacing
        mesh = grid.contour([0.5])
        
        # 2. 骨骼网格：保体积微松弛平滑
        print(f"    -> [Unity] 智能平滑 (去噪点且保体积)...")
        mesh = mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True
        )
        
        # 3. 骨骼网格：特征保护法线与减面
        print("    -> [Unity] 烘焙高质量顶点法线...")
        mesh = mesh.compute_normals(
            point_normals=True, 
            split_vertices=True, 
            feature_angle=60.0,
            auto_orient_normals=True
        )
        
        if self.config.get("decimate_ratio", 0) > 0:
            print(f"    -> [减面] 拓扑保护减面...")
            mesh = mesh.decimate(self.config["decimate_ratio"])
        
        # 【修改点1】：骨骼不需要热力图，直接删除 UV 赋值，跳过报错，加快导出速度！
        meshes['bone'] = mesh
        
        # 4. 肿瘤网格：热力图 UV 隐写
        if pet_data is not None:
            if not hasattr(self, 'tumor_info'):
                self.extract_tumors_advanced(pet_data, ct_data=None, spacing=spacing, bone_mask=bone_mask)
            
            if hasattr(self, 'tumor_info') and self.tumor_info and 'masks' in self.tumor_info:
                print("    -> [Mesh] 正在处理肿瘤热力图 UV 网格...")
                tumor_meshes = []
                for t_id, t_mask in self.tumor_info['masks'].items():
                    # 肿瘤同样使用亚像素平滑
                    t_float = ndimage.gaussian_filter(t_mask.astype(np.float32), sigma=0.8)
                    t_grid = pv.wrap(t_float)
                    t_grid.spacing = spacing
                    t_mesh = t_grid.contour([0.5]).smooth(n_iter=50, relaxation_factor=0.02)
                    tumor_meshes.append(t_mesh)
                
                if tumor_meshes:
                    combined_tumor = tumor_meshes[0]
                    if len(tumor_meshes) > 1:
                        for tm in tumor_meshes[1:]:
                            combined_tumor = combined_tumor.merge(tm)
                    
                    combined_tumor = combined_tumor.compute_normals(point_normals=True, auto_orient_normals=True)
                    
                    # 提取 PET 强度
                    pts = combined_tumor.points
                    indices = np.column_stack((pts[:, 0]/spacing[0], pts[:, 1]/spacing[1], pts[:, 2]/spacing[2]))
                    pet_vals = ndimage.map_coordinates(pet_data, indices.T, order=1)
                    p_min, p_max = np.min(pet_vals), np.percentile(pet_vals, 95)
                    norm_vals = np.clip((pet_vals - p_min) / (p_max - p_min + 1e-6), 0.0, 1.0)
                    
                    # 【修改点2】：暴力绕开 PyVista 属性锁，直接把 UV 数组注入到底层 point_data 字典中！
                    uv_array = np.column_stack((norm_vals, np.zeros_like(norm_vals)))
                    combined_tumor.point_data["Texture Coordinates"] = uv_array
                    # 双重保险：同时存一份标量数据，防止导出丢失
                    combined_tumor.point_data["SUV_Heatmap"] = norm_vals
                    
                    meshes['tumor'] = combined_tumor
        
        # 5. 医生标注网格 (Label)
        if label_data is not None and np.any(label_data):
            lbl_float = ndimage.gaussian_filter(label_data.astype(np.float32), sigma=0.8)
            lbl_grid = pv.wrap(lbl_float)
            lbl_grid.spacing = spacing
            lbl_mesh = lbl_grid.contour([0.5]).smooth(n_iter=30, relaxation_factor=0.03)
            lbl_mesh = lbl_mesh.compute_normals(point_normals=True, auto_orient_normals=True)
            # 【修改点3】：医生标注是纯色的，也不需要 UV，直接删掉赋值，完美避开报错！
            meshes['label'] = lbl_mesh
            
        return meshes
    
    def export_models(self, meshes, output_dir, case_name):
        """统一导出为标准OBJ格式"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for mesh_name, mesh in meshes.items():
            file_path = os.path.join(output_dir, f"{case_name}_V2_{mesh_name}.obj")
            
            if 'Normals' not in mesh.point_data:
                print(f"    -> [警告] {mesh_name} 缺少法线，重新计算")
                mesh = mesh.compute_normals(point_normals=True)
            
            try:
                mesh.save(file_path)
                print(f"✓ 导出成功: {file_path} ({mesh.n_cells:,} 面)")
            except Exception as e:
                print(f"导出失败: {mesh_name} - {e}")
    
    # --- 辅助滤镜库 ---
    def _anisotropic_diffusion(self, data, iterations=3, kappa=50):
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
    
    def _calculate_tumor_confidence(self, features, global_suv_max):
        score = 0.0
        suv_ratio = features['suv_peak'] / global_suv_max
        score += suv_ratio * 0.5
        vol_score = min(features['volume'] / 10.0, 1.0)
        score += vol_score * 0.3
        if 'in_bone_ratio' in features:
            score += features['in_bone_ratio'] * 0.2
        return min(score, 1.0)