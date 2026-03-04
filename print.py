#!/usr/bin/env python
"""
print_all_v2_labels.py

打印 TotalSegmentator V2 中所有标签的 ID 和名称。
确保你已安装 TotalSegmentator 2.x。
"""

from totalsegmentator.map_to_binary import class_map

def print_all_labels():
    # 获取 V2 的 total 任务映射（优先使用 "total_v2"，如果没有则用 "total"）
    cmap = class_map.get("total_v2", class_map.get("total", {}))
    if not cmap:
        print("错误：无法获取 TotalSegmentator V2 的标签映射。")
        return

    print("\nTotalSegmentator V2 所有标签 (ID : 名称)")
    print("-" * 60)
    for class_id, name in sorted(cmap.items()):
        print(f"{class_id:3d} : {name}")
    print(f"\n总数：{len(cmap)} 个标签")

if __name__ == "__main__":
    print_all_labels()