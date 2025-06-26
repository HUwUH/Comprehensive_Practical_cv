from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

def plot_multi_model_pr_curve(models_data, save_dir=Path("multi_model_pr_curve.png"), on_plot=None):
    """
    绘制多个模型的PR曲线对比图
    
    Args:
        models_data: 字典，格式为 {model_name: {"px": px, "py": py, "ap": ap}}
        save_dir: 保存路径
        on_plot: 回调函数
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), tight_layout=True)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (model_name, data) in enumerate(models_data.items()):
        px = data["px"]
        py = data["py"]
        ap = data["ap"]
        
        color = colors[i % len(colors)]
        
        # 如果py是多维数组，计算平均值
        if isinstance(py, np.ndarray) and py.ndim > 1:
            # 如果是(1, n)的形状，取第一行
            if py.shape[0] == 1:
                py_mean = py[0]
            # 如果是(n, 1)的形状，取第一列
            elif py.shape[1] == 1:
                py_mean = py[:, 0]
            # 其他情况，对于PR曲线，通常是(n_thresholds, n_classes)的形状
            # 我们需要计算所有类别的平均值
            else:
                py_mean = py.mean(axis=1)
        else:
            py_mean = py
            
        # 计算mAP
        if isinstance(ap, np.ndarray):
            if ap.ndim > 1:
                # 如果是(1, n)的形状，取第一行
                if ap.shape[0] == 1:
                    map_score = ap[0].mean()
                # 如果是(n, 1)的形状，取第一列
                elif ap.shape[1] == 1:
                    map_score = ap[:, 0].mean()
                # 其他情况
                else:
                    map_score = ap[:, 0].mean() if ap.shape[1] > 0 else ap.mean()
            else:
                map_score = ap.mean()
        else:
            map_score = ap
            
        ax.plot(px, py_mean, linewidth=2, color=color, 
                label=f"{model_name} (mAP@0.5: {map_score:.3f})")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Multi-Model Precision-Recall Curve Comparison")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir, dpi=250, bbox_inches='tight')
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

def plot_multi_model_mc_curve(models_data, save_dir=Path("multi_model_mc_curve.png"), 
                              xlabel="Confidence", ylabel="Metric", on_plot=None):
    """
    绘制多个模型的metric-confidence曲线对比图
    
    Args:
        models_data: 字典，格式为 {model_name: {"px": px, "py": py}}
        save_dir: 保存路径
        xlabel: x轴标签
        ylabel: y轴标签
        on_plot: 回调函数
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), tight_layout=True)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (model_name, data) in enumerate(models_data.items()):
        px = data["px"]
        py = data["py"]
        
        color = colors[i % len(colors)]
        
        # 处理多维数组
        if isinstance(py, np.ndarray) and py.ndim > 1:
            # 如果是(1, n)的形状，取第一行
            if py.shape[0] == 1:
                py_mean = py[0]
            # 如果是(n, 1)的形状，取第一列  
            elif py.shape[1] == 1:
                py_mean = py[:, 0]
            # 其他情况计算平均值
            else:
                py_mean = py.mean(0) if py.shape[0] > py.shape[1] else py.mean(1)
        else:
            py_mean = py
            
        # 确保px和py_mean长度一致
        if len(py_mean) != len(px):
            # 如果长度不一致，调整py_mean的长度
            if len(py_mean) > len(px):
                py_mean = py_mean[:len(px)]
            else:
                # 使用插值来匹配长度
                py_mean = np.interp(np.linspace(0, 1, len(px)), 
                                   np.linspace(0, 1, len(py_mean)), py_mean)
            
        # 平滑处理
        y_smooth = smooth(py_mean, 0.05)
        
        # 确保平滑后的数据与px长度一致
        if len(y_smooth) != len(px):
            y_smooth = np.interp(np.linspace(0, 1, len(px)), 
                               np.linspace(0, 1, len(y_smooth)), y_smooth)
        max_val = y_smooth.max()
        max_idx = y_smooth.argmax()
        
        ax.plot(px, y_smooth, linewidth=2, color=color,
                label=f"{model_name} ({max_val:.3f} at {px[max_idx]:.3f})")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"Multi-Model {ylabel}-Confidence Curve Comparison")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir, dpi=250, bbox_inches='tight')
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

def compare_multiple_models(models_data, save_dir=Path("."), prefix="comparison_", on_plot=None):
    """
    比较多个模型的性能，生成所有对比图
    
    Args:
        models_data: 字典，格式为 {
            model_name: {
                "x": x,
                "prec_values": prec_values,
                "ap": ap,
                "f1_curve": f1_curve,
                "p_curve": p_curve,
                "r_curve": r_curve
            }
        }
        save_dir: 保存目录
        prefix: 文件名前缀
        on_plot: 回调函数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # PR曲线对比
    pr_data = {}
    for model_name, data in models_data.items():
        pr_data[model_name] = {
            "px": data["x"],
            "py": data["prec_values"],
            "ap": data["ap"]
        }
    plot_multi_model_pr_curve(pr_data, save_dir / f"{prefix}PR_curve.png", on_plot)
    
    # F1曲线对比
    f1_data = {}
    for model_name, data in models_data.items():
        f1_data[model_name] = {
            "px": data["x"],
            "py": data["f1_curve"]
        }
    plot_multi_model_mc_curve(f1_data, save_dir / f"{prefix}F1_curve.png", 
                              ylabel="F1", on_plot=on_plot)
    
    # Precision曲线对比
    p_data = {}
    for model_name, data in models_data.items():
        p_data[model_name] = {
            "px": data["x"],
            "py": data["p_curve"]
        }
    plot_multi_model_mc_curve(p_data, save_dir / f"{prefix}P_curve.png", 
                              ylabel="Precision", on_plot=on_plot)
    
    # Recall曲线对比
    r_data = {}
    for model_name, data in models_data.items():
        r_data[model_name] = {
            "px": data["x"],
            "py": data["r_curve"]
        }
    plot_multi_model_mc_curve(r_data, save_dir / f"{prefix}R_curve.png", 
                              ylabel="Recall", on_plot=on_plot)

def main(x, prec_values, ap, f1_curve, p_curve, r_curve, save_dir=Path("."), names={}, prefix="", on_plot=None):
    plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
    plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
    plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
    plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

# 使用示例
if __name__ == "__main__":
    # 示例：比较多个模型
    # models_data = {
    #     "YOLOv8n": {
    #         "x": x1,
    #         "prec_values": prec_values1,
    #         "ap": ap1,
    #         "f1_curve": f1_curve1,
    #         "p_curve": p_curve1,
    #         "r_curve": r_curve1
    #     },
    #     "YOLOv8s": {
    #         "x": x2,
    #         "prec_values": prec_values2,
    #         "ap": ap2,
    #         "f1_curve": f1_curve2,
    #         "p_curve": p_curve2,
    #         "r_curve": r_curve2
    #     }
    # }
    # compare_multiple_models(models_data, save_dir="./comparison_results/")
    pass

def load_model_data_from_folder(folder_path):
    """
    从文件夹中加载模型数据
    
    Args:
        folder_path: 包含.npy文件的文件夹路径
        
    Returns:
        字典包含加载的数据，如果文件不存在则返回None
    """
    folder_path = Path(folder_path)
    
    # 定义需要加载的文件名
    files_to_load = {
        'x': 'x.npy',
        'prec_values': 'prec_values.npy', 
        'ap': 'ap.npy',
        'f1_curve': 'f1_curve.npy',
        'p_curve': 'p_curve.npy', 
        'r_curve': 'r_curve.npy'
    }
    
    # 可能的文件名变体
    alternative_names = {
        'x': ['x.npy', 'confidence.npy', 'conf.npy'],
        'prec_values': ['prec_values.npy', 'precision_values.npy', 'pr_values.npy'],
        'ap': ['ap.npy', 'average_precision.npy', 'mAP.npy'],
        'f1_curve': ['f1_curve.npy', 'f1.npy'],
        'p_curve': ['p_curve.npy', 'precision_curve.npy', 'p.npy'],
        'r_curve': ['r_curve.npy', 'recall_curve.npy', 'r.npy']
    }
    
    data = {}
    
    for key, filename in files_to_load.items():
        file_path = folder_path / filename
        
        # 如果主文件名不存在，尝试备选文件名
        if not file_path.exists():
            found = False
            for alt_name in alternative_names.get(key, []):
                alt_path = folder_path / alt_name
                if alt_path.exists():
                    file_path = alt_path
                    found = True
                    break
            
            if not found:
                print(f"警告: 在 {folder_path} 中找不到 {key} 数据文件")
                continue
        
        try:
            data[key] = np.load(file_path)
            print(f"成功加载: {file_path}")
        except Exception as e:
            print(f"加载失败 {file_path}: {e}")
            continue
    
    return data if data else None

def plot_models_from_folders(models_folders, save_dir=Path("./comparison_results/"), prefix="models_", on_plot=None):
    """
    从多个文件夹加载模型数据并生成对比图
    
    Args:
        models_folders: 字典，格式为 {model_name: folder_path}
        save_dir: 保存目录
        prefix: 文件名前缀
        on_plot: 回调函数
        
    Example:
        models_folders = {
            "YOLOv8n": "./runs/detect/yolov8n/",
            "YOLOv8s": "./runs/detect/yolov8s/", 
            "YOLOv8m": "./runs/detect/yolov8m/"
        }
        plot_models_from_folders(models_folders)
    """
    models_data = {}
    
    # 加载所有模型数据
    for model_name, folder_path in models_folders.items():
        print(f"\n正在加载模型 {model_name} 的数据...")
        data = load_model_data_from_folder(folder_path)
        
        if data is None or len(data) == 0:
            print(f"跳过模型 {model_name}: 无法加载数据")
            continue
            
        # 检查必需的数据是否存在
        required_keys = ['x', 'f1_curve', 'p_curve', 'r_curve']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print(f"跳过模型 {model_name}: 缺少必需数据 {missing_keys}")
            continue
            
        models_data[model_name] = data
        print(f"模型 {model_name} 数据加载完成")
    
    if not models_data:
        print("错误: 没有成功加载任何模型数据")
        return
    
    print(f"\n开始生成 {len(models_data)} 个模型的对比图...")
    
    # 生成对比图
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # PR曲线对比 (如果有prec_values和ap数据)
    pr_models = {}
    for model_name, data in models_data.items():
        if 'prec_values' in data and 'ap' in data:
            pr_models[model_name] = {
                "px": data["x"],
                "py": data["prec_values"],
                "ap": data["ap"]
            }
    
    if pr_models:
        plot_multi_model_pr_curve(pr_models, save_dir / f"{prefix}PR_curve.png", on_plot)
        print(f"PR曲线对比图已保存")
    
    # F1曲线对比
    f1_data = {}
    for model_name, data in models_data.items():
        f1_data[model_name] = {
            "px": data["x"],
            "py": data["f1_curve"]
        }
    plot_multi_model_mc_curve(f1_data, save_dir / f"{prefix}F1_curve.png", 
                              ylabel="F1", on_plot=on_plot)
    print(f"F1曲线对比图已保存")
    
    # Precision曲线对比
    p_data = {}
    for model_name, data in models_data.items():
        p_data[model_name] = {
            "px": data["x"],
            "py": data["p_curve"]
        }
    plot_multi_model_mc_curve(p_data, save_dir / f"{prefix}P_curve.png", 
                              ylabel="Precision", on_plot=on_plot)
    print(f"Precision曲线对比图已保存")
    
    # Recall曲线对比
    r_data = {}
    for model_name, data in models_data.items():
        r_data[model_name] = {
            "px": data["x"],
            "py": data["r_curve"]
        }
    plot_multi_model_mc_curve(r_data, save_dir / f"{prefix}R_curve.png", 
                              ylabel="Recall", on_plot=on_plot)
    print(f"Recall曲线对比图已保存")
    
    print(f"\n所有对比图已保存到: {save_dir}")
    return models_data

def auto_discover_models(base_dir, pattern="*/", exclude_dirs=None):
    """
    自动发现指定目录下的模型文件夹
    
    Args:
        base_dir: 基础目录路径
        pattern: 搜索模式，默认为所有子目录
        exclude_dirs: 要排除的目录名列表
        
    Returns:
        字典 {model_name: folder_path}
    """
    base_dir = Path(base_dir)
    exclude_dirs = exclude_dirs or ['__pycache__', '.git', '.vscode', 'wandb']
    
    models_folders = {}
    
    # 搜索匹配的目录
    for folder in base_dir.glob(pattern):
        if folder.is_dir() and folder.name not in exclude_dirs:
            # 检查是否包含.npy文件
            npy_files = list(folder.glob("*.npy"))
            if npy_files:
                models_folders[folder.name] = str(folder)
                
    return models_folders