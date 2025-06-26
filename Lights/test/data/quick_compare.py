"""
简单的模型比较脚本
直接修改models_folders字典中的路径，然后运行即可
"""

from show import plot_models_from_folders
from pathlib import Path

def main():
    """
    主函数：比较你的模型
    请修改下面的models_folders字典，设置为你的实际模型文件夹路径
    """
    
    # ===========================================
    # 修改这里：设置你的模型文件夹路径
    # ===========================================
    models_folders = {
        # "Yolo11n1": r"D:\MyFile\qq_3045834499\yolo11-chest\42_demo\runs\detect\val",
        "Yolo11n2": r"D:\MyFile\qq_3045834499\yolo11-chest\42_demo\runs\detect\val2",
        # "Yolo11s1": r"D:\MyFile\qq_3045834499\yolo11-chest\42_demo\runs\detect\val3",
        "Yolo11s2": r"D:\MyFile\qq_3045834499\yolo11-chest\42_demo\runs\detect\val4", 
        "Yolo11l1": r"D:\MyFile\qq_3045834499\yolo11-chest\42_demo\runs\detect\val5",
        "Yolo11l2": r"D:\MyFile\qq_3045834499\yolo11-chest\42_demo\runs\detect\val6",
        # "模型3": "./path/to/model3/",  # 替换为你的模型3文件夹路径
        # 添加更多模型...
    }
    
    # ===========================================
    # 可选：修改保存设置
    # ===========================================
    save_directory = "./results/"  # 结果保存目录
    file_prefix = ""  # 文件名前缀
    
    # ===========================================
    # 运行比较
    # ===========================================
    print("开始比较模型性能...")
    print(f"模型列表: {list(models_folders.keys())}")
    
    try:
        plot_models_from_folders(
            models_folders=models_folders,
            save_dir=save_directory,
            prefix=file_prefix
        )
        
        print(f"\n✅ 成功！对比图已保存到: {save_directory}")
        print("\n生成的文件:")
        print(f"- {file_prefix}F1_curve.png")
        print(f"- {file_prefix}P_curve.png") 
        print(f"- {file_prefix}R_curve.png")
        print(f"- {file_prefix}PR_curve.png (如果有PR数据)")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请检查:")
        print("1. 文件夹路径是否正确")
        print("2. 文件夹中是否包含必需的.npy文件")
        print("3. 数据文件格式是否正确")

if __name__ == "__main__":
    main()
