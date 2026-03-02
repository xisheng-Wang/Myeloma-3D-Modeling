# BoneExtractor_V1.py - 传统增强版（借鉴V3迟滞阈值+形态学）
# 策略：纯CT阈值流，无AI，适合对比V2的AI混合流
import numpy as np
from scipy import ndimage
from skimage import morphology
import time
import pyvista as pv


class BoneExtractorV1:
    """传统增强版骨骼提取（V3迟滞阈值+形态学优化，无AI）"""

    def __init__(self, config=None):
        self.config = {
            "gaussian_sigma": 1.0,
            "bed_removal_radius": 0.48,
            "bone_high_threshold": 220,
            "bone_low_threshold": 80,
            "morph_kernel_size": 3,
            "smooth_iter": 100,
            "smooth_relax": 0.05,
            "decimate_ratio": 0.1,
        }
        if config:
            self.config.update(config)

    def extract_bone(self, ct_data, spacing=None):
        """骨骼提取：V3风格迟滞阈值 + 形态学闭/开运算"""
        print(">>> [V1 传统增强] 迟滞阈值+形态学提取 <<<")
        t0 = time.time()

        # 1. 高斯平滑
        ct_smooth = ndimage.gaussian_filter(
            ct_data, sigma=self.config["gaussian_sigma"]
        )

        # 2. 圆柱体ROI掩码
        d, h, w = ct_smooth.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask_roi = ((x - cx) ** 2 + (y - cy) ** 2) < (
            w * self.config["bed_removal_radius"]
        ) ** 2
        mask_roi = mask_roi[np.newaxis, :, :]

        # 3. 迟滞阈值（V3核心）
        mask_high = (ct_smooth > self.config["bone_high_threshold"]) & mask_roi
        mask_low = (ct_smooth > self.config["bone_low_threshold"]) & mask_roi

        # 4. 连通性重建
        labeled_low, _ = ndimage.label(mask_low)
        valid_labels = np.unique(labeled_low[mask_high])
        valid_labels = valid_labels[valid_labels != 0]
        bone_mask = (
            np.isin(labeled_low, valid_labels)
            if len(valid_labels) > 0
            else mask_high
        )

        # 5. 形态学闭运算（修复断裂）
        struct_elem = morphology.ball(self.config["morph_kernel_size"])
        bone_closed = morphology.binary_closing(bone_mask, struct_elem)

        # 6. 开运算去噪
        bone_opened = morphology.binary_opening(bone_closed, morphology.ball(1))

        # 7. 连通域筛选
        labeled_final, num_final = ndimage.label(bone_opened)
        if num_final > 0:
            slices = ndimage.find_objects(labeled_final)
            volumes = []
            for i, s in enumerate(slices):
                if s is not None:
                    vol = np.sum(labeled_final[s] == (i + 1))
                    volumes.append(vol)

            sorted_indices = np.argsort(volumes)[::-1]
            final_bone_mask = np.zeros_like(bone_opened, dtype=bool)
            max_vol = volumes[sorted_indices[0]] if volumes else 0

            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i] + 1
                vol = volumes[sorted_indices[i]]
                if vol > max_vol * 0.05:
                    final_bone_mask |= labeled_final == idx
        else:
            final_bone_mask = bone_opened

        # 8. 填充空洞
        final_bone_mask = ndimage.binary_fill_holes(final_bone_mask)

        print(f"V1 骨骼提取完成. 耗时: {time.time()-t0:.2f}s")
        return final_bone_mask

    def _extract_tumor_simple(self, pet_data, bone_mask=None):
        """简单骨髓瘤提取：PET阈值 + 骨骼空间约束"""
        if pet_data is None:
            return None
        threshold = np.max(pet_data) * 0.4
        tumor_mask = pet_data > threshold
        if bone_mask is not None:
            bone_dilated = ndimage.binary_dilation(bone_mask, iterations=5)
            tumor_mask = tumor_mask & bone_dilated
        if not np.any(tumor_mask):
            return None
        return tumor_mask

    def create_mesh(self, bone_mask, spacing, label_data=None, pet_data=None):
        """生成Unity友好OBJ网格"""
        meshes = {}

        # 骨骼网格
        bone_grid = pv.wrap(bone_mask.astype(np.uint8))
        bone_grid.spacing = spacing
        bone_mesh = bone_grid.contour([0.5])
        bone_mesh = bone_mesh.smooth(
            n_iter=self.config["smooth_iter"],
            relaxation_factor=self.config["smooth_relax"],
            boundary_smoothing=True,
        )
        bone_mesh = bone_mesh.compute_normals(
            point_normals=True, auto_orient_normals=True
        )
        bone_mesh = bone_mesh.decimate(self.config["decimate_ratio"])
        meshes["bone"] = bone_mesh

        # 骨髓瘤网格（PET简单阈值）
        tumor_mask = self._extract_tumor_simple(pet_data, bone_mask)
        if tumor_mask is not None:
            tumor_grid = pv.wrap(tumor_mask.astype(np.uint8))
            tumor_grid.spacing = spacing
            tumor_mesh = tumor_grid.contour([0.5])
            tumor_mesh = tumor_mesh.smooth(n_iter=20, relaxation_factor=0.1)
            tumor_mesh = tumor_mesh.compute_normals(
                point_normals=True, auto_orient_normals=True
            )
            meshes["tumor"] = tumor_mesh

        # 医生标注网格
        if label_data is not None and np.any(label_data):
            lbl_grid = pv.wrap(label_data.astype(np.uint8))
            lbl_grid.spacing = spacing
            label_mesh = lbl_grid.contour([0.5])
            label_mesh = label_mesh.smooth(n_iter=20, relaxation_factor=0.1)
            label_mesh = label_mesh.compute_normals(
                point_normals=True, auto_orient_normals=True
            )
            meshes["label"] = label_mesh

        return meshes

    def export_models(self, meshes, output_dir, case_name):
        """导出OBJ（Unity可用）"""
        import os

        os.makedirs(output_dir, exist_ok=True)
        for mesh_name, mesh in meshes.items():
            path = os.path.join(output_dir, f"{case_name}_V1_{mesh_name}.obj")
            if "Normals" not in mesh.point_data:
                mesh = mesh.compute_normals(point_normals=True)
            mesh.save(path)
            print(f"导出: {path}")
