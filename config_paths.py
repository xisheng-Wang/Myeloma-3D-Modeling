# config_paths.py - 路径配置
import os

class PathConfig:
    """路径配置类 - 用户只需修改这里"""
    
    # ============ 批量处理配置 ============
    # 方式1: 单个病例处理
    SINGLE_CASE = {
        "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000215239/E000215239__CT.nii/E000215239__CT.nii",
        "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000215239/E000215239__PT.nii/E000215239__PT.nii",
        "label_path": r"D:/Project/医学影像三维建模/medical picture/E000215239/E000215239_2.5CK.nii/E000215239_2.5CK.nii",
        "case_name": "E000215239"
    }
    
    # 方式2: 批量处理（多个病例）
    BATCH_CASES = [
        # === 现有病例 ===
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000215239/E000215239__CT.nii/E000215239__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000215239/E000215239__PT.nii/E000215239__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000215239/E000215239_2.5CK.nii/E000215239_2.5CK.nii",
            "case_name": "E000215239"
        },
        # === 新增病例 (来自截图) ===
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000183855/E000183855__CT.nii/E000183855__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000183855/E000183855__PT.nii/E000183855__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000183855/E000183855_2.5CK.nii/E000183855_2.5CK.nii",
            "case_name": "E000183855"
        },
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000188870/E000188870__CT.nii/E000188870__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000188870/E000188870__PT.nii/E000188870__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000188870/E000188870_2.5CK.nii/E000188870_2.5CK.nii",
            "case_name": "E000188870"
        },
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000189678/E000189678__CT.nii/E000189678__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000189678/E000189678__PT.nii/E000189678__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000189678/E000189678_2.5CK.nii/E000189678_2.5CK.nii",
            "case_name": "E000189678"
        },
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000189861/E000189861__CT.nii/E000189861__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000189861/E000189861__PT.nii/E000189861__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000189861/E000189861_2.5CK.nii/E000189861_2.5CK.nii",
            "case_name": "E000189861"
        },
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000192635/E000192635__CT.nii/E000192635__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000192635/E000192635__PT.nii/E000192635__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000192635/E000192635_2.5CK.nii/E000192635_2.5CK.nii",
            "case_name": "E000192635"
        },
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000193196/E000193196__CT.nii/E000193196__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000193196/E000193196__PT.nii/E000193196__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000193196/E000193196_2.5CK.nii/E000193196_2.5CK.nii",
            "case_name": "E000193196"
        },
        {
            "ct_path": r"D:/Project/医学影像三维建模/medical picture/E000198507/E000198507__CT.nii/E000198507__CT.nii",
            "pet_path": r"D:/Project/医学影像三维建模/medical picture/E000198507/E000198507__PT.nii/E000198507__PT.nii",
            "label_path": r"D:/Project/医学影像三维建模/medical picture/E000198507/E000198507_2.5CK.nii/E000198507_2.5CK.nii",
            "case_name": "E000198507"
        },
    ]
    
    # ============ 输出目录配置 ============
    # 输出根目录
    OUTPUT_ROOT = r"D:/Project/医学影像三维建模/3D_Models_Batch"
    
    # 各版本算法输出子目录
    OUTPUT_PATHS = {
        "V1": os.path.join(OUTPUT_ROOT, "V1_Original"),
        "V2": os.path.join(OUTPUT_ROOT, "V2_Optimized"),
        "V3": os.path.join(OUTPUT_ROOT, "V3_Unity"),
        "AI": os.path.join(OUTPUT_ROOT, "AI_Enhanced"),
        "AI_TEMP": os.path.join(OUTPUT_ROOT, "AI_Temp")  # AI临时文件目录
    }
    
    # ============ 算法选择配置 ============
    # 要运行的算法版本列表
    # 修改这里来选择要运行的算法
    
    # 只运行AI算法（你想要的）
    ALGORITHMS_TO_RUN = ["AI"]
    
    # 运行所有算法
    # ALGORITHMS_TO_RUN = ["V1", "V2", "V3", "AI"]
    
    @staticmethod
    def check_paths():
        """检查路径是否存在"""
        import os
        
        # 检查输出目录
        for dir_name, dir_path in PathConfig.OUTPUT_PATHS.items():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"创建目录: {dir_path}")
        
        return True