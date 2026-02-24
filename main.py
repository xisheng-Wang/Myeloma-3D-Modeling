# main.py - 主程序（批量处理）
import os
import time
import traceback
from datetime import datetime

# 导入配置
from config_paths import PathConfig

# 导入算法类
from BoneExtractor_V1 import BoneExtractorV1
from BoneExtractor_V2 import BoneExtractorV2
from BoneExtractor_V3 import BoneExtractorV3
from BoneExtractor_AI import BoneExtractorAI

# 导入工具函数
from utils import load_ct_data, load_and_align_data

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self):
        self.statistics = []  # 统计信息
        
    def process_single_case(self, case_info, algorithms=None):
        """处理单个病例"""
        case_name = case_info['case_name']
        print(f"开始处理病例: {case_name}")
        
        # 加载CT数据
        try:
            ct_data, ct_affine, spacing = load_ct_data(case_info['ct_path'])
            print(f"CT数据加载完成: {ct_data.shape}")
        except Exception as e:
            print(f"加载CT数据失败: {e}")
            return False
        
        # 加载PET数据（如果存在）
        pet_data = None
        if 'pet_path' in case_info and case_info['pet_path']:
            pet_data = load_and_align_data(
                case_info['pet_path'], 
                ct_data.shape, 
                ct_affine, 
                interpolation=1
            )
            if pet_data is not None:
                print(f"PET数据加载完成")
        
        # 加载Label数据（如果存在）
        label_data = None
        if 'label_path' in case_info and case_info['label_path']:
            label_data = load_and_align_data(
                case_info['label_path'], 
                ct_data.shape, 
                ct_affine, 
                interpolation=0
            )
            if label_data is not None:
                label_data = (label_data > 0.5).astype(bool)
                print(f"Label数据加载完成")
        
        # 如果没有指定算法，使用配置中的所有算法
        if algorithms is None:
            algorithms = PathConfig.ALGORITHMS_TO_RUN
        
        # 运行指定的算法
        for algo in algorithms:
            output_dir = PathConfig.OUTPUT_PATHS.get(algo)
            if not output_dir:
                print(f"警告: 算法 {algo} 的输出目录未配置，跳过")
                continue
            
            try:
                print(f"\n{'='*60}")
                print(f"运行算法: {algo}")
                print(f"{'='*60}")
                
                # 选择算法
                if algo == "V1":
                    extractor = BoneExtractorV1()
                    bone_mask = extractor.extract_bone(ct_data, spacing)
                    meshes = extractor.create_mesh(bone_mask, spacing, label_data, pet_data)
                    extractor.export_models(meshes, output_dir, case_name)
                    
                elif algo == "V2":
                    extractor = BoneExtractorV2()
                    bone_mask = extractor.extract_bone(ct_data, spacing)
                    meshes = extractor.create_mesh(bone_mask, spacing, label_data, pet_data)
                    extractor.export_models(meshes, output_dir, case_name)
                    
                elif algo == "V3":
                    extractor = BoneExtractorV3()
                    bone_mask = extractor.extract_bone(ct_data, spacing)
                    meshes = extractor.create_mesh(bone_mask, spacing, label_data, pet_data)
                    extractor.export_models(meshes, output_dir, case_name)
                    
                elif algo == "AI":
                    extractor = BoneExtractorAI()
                    
                    # AI算法需要额外的AI分割步骤
                    ai_temp_dir = PathConfig.OUTPUT_PATHS["AI_TEMP"]
                    ai_case_dir = os.path.join(ai_temp_dir, case_name)
                    
                    print(f"AI临时目录: {ai_case_dir}")
                    ai_output_path = extractor.run_ai_segmentation(
                        ct_data, ct_affine, 
                        ai_case_dir
                    )
                    
                    bone_mask = extractor.extract_bone(
                        ct_data, ct_affine, spacing, ai_output_path
                    )
                    meshes = extractor.create_mesh(bone_mask, spacing, label_data, pet_data)
                    extractor.export_models(meshes, output_dir, case_name)
                    
                else:
                    print(f"警告: 未知算法: {algo}，跳过")
                    continue
                
                print(f"算法 {algo} 处理完成")
                
            except Exception as e:
                print(f"算法 {algo} 处理失败: {e}")
                traceback.print_exc()
        
        return True
    
    def process_batch(self, batch_cases=None, algorithms=None):
        """批量处理多个病例"""
        if batch_cases is None:
            batch_cases = PathConfig.BATCH_CASES
        
        if not batch_cases:
            print("没有病例需要处理")
            return False
        
        # 检查并创建输出目录
        PathConfig.check_paths()
        
        total_cases = len(batch_cases)
        successful_cases = 0
        
        print(f"开始批量处理，共 {total_cases} 个病例")
        print(f"运行的算法: {algorithms or PathConfig.ALGORITHMS_TO_RUN}")
        
        start_time = time.time()
        
        for i, case_info in enumerate(batch_cases, 1):
            print(f"\n{'#'*70}")
            print(f"处理病例 {i}/{total_cases}: {case_info.get('case_name', f'Case_{i}')}")
            print(f"{'#'*70}")
            
            try:
                if self.process_single_case(case_info, algorithms):
                    successful_cases += 1
                    print(f"病例 {i} 处理成功")
                else:
                    print(f"病例 {i} 处理失败")
                    
            except Exception as e:
                print(f"处理病例 {i} 时发生错误: {e}")
                traceback.print_exc()
        
        # 输出处理摘要
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("批量处理完成!")
        print("="*70)
        print(f"总病例数: {total_cases}")
        print(f"成功处理: {successful_cases}")
        print(f"处理失败: {total_cases - successful_cases}")
        print(f"成功率: {successful_cases/total_cases*100:.1f}%" if total_cases > 0 else "0%")
        print(f"总耗时: {duration:.1f} 秒")
        print("="*70)
        
        return successful_cases

def main():
    """主函数"""
    print("\n" + "="*70)
    print("    医学影像三维重建批量处理系统")
    print("="*70)
    
    # 显示当前配置
    print("当前配置:")
    print(f"  输出根目录: {PathConfig.OUTPUT_ROOT}")
    print(f"  启用的算法: {PathConfig.ALGORITHMS_TO_RUN}")
    
    if PathConfig.BATCH_CASES:
        print(f"  批量病例数: {len(PathConfig.BATCH_CASES)}")
        for i, case in enumerate(PathConfig.BATCH_CASES, 1):
            print(f"    病例{i}: {case['case_name']}")
    else:
        print(f"  单个病例: {PathConfig.SINGLE_CASE['case_name']}")
    
    print("="*70)
    
    # 创建处理器
    processor = BatchProcessor()
    
    try:
        while True:
            print("\n选择操作:")
            print("  1. 处理单个病例")
            print("  2. 批量处理所有病例")
            print("  3. 配置运行算法")
            print("  4. 退出程序")
            
            mode_choice = input("\n请输入选择 (1-4): ").strip()
            
            if mode_choice == "1":
                # 处理单个病例
                print("\n选择要运行的算法:")
                print("  V1 - 原始版本")
                print("  V2 - 优化版本")
                print("  V3 - Unity版本")
                print("  AI - AI增强版本")
                print("  或输入 'all' 运行所有算法")
                
                algo_input = input("\n请输入算法代码 (例如: AI 或 V1,V2,V3): ").strip()
                
                if algo_input.lower() == 'all':
                    algorithms_to_run = PathConfig.ALGORITHMS_TO_RUN
                else:
                    algorithms_to_run = [a.strip() for a in algo_input.split(',')]
                
                print(f"\n将运行算法: {algorithms_to_run}")
                
                confirm = input("\n确认开始处理? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    processor.process_single_case(
                        PathConfig.SINGLE_CASE, 
                        algorithms=algorithms_to_run
                    )
                else:
                    print("已取消处理")
                    
            elif mode_choice == "2":
                # 批量处理
                confirm = input("\n确认批量处理所有病例? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    processor.process_batch()
                else:
                    print("已取消处理")
                    
            elif mode_choice == "3":
                # 配置运行算法
                print(f"\n当前启用的算法: {PathConfig.ALGORITHMS_TO_RUN}")
                print("\n请输入要启用的算法代码 (用逗号分隔，例如: V2,AI):")
                algo_input = input("算法代码: ").strip()
                
                if algo_input:
                    new_algorithms = [a.strip() for a in algo_input.split(',')]
                    valid_algorithms = ["V1", "V2", "V3", "AI"]
                    valid_input = True
                    
                    for algo in new_algorithms:
                        if algo not in valid_algorithms:
                            print(f"错误: 未知算法 '{algo}'")
                            valid_input = False
                            break
                    
                    if valid_input:
                        PathConfig.ALGORITHMS_TO_RUN = new_algorithms
                        print(f"已更新算法配置: {PathConfig.ALGORITHMS_TO_RUN}")
                    else:
                        print("配置未更新，请使用有效的算法代码")
                else:
                    print("输入为空，配置未更新")
                    
            elif mode_choice == "4":
                print("程序退出")
                break
                
            else:
                print("无效选择，请重新输入")
                
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()