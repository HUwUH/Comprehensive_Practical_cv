from ultralytics import YOLO

if __name__ == '__main__':
    # 设置环境变量，解决多线程加载数据时可能出现的错误
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    model = YOLO(r"D:\Programming\Pycharm\Comprehensive_Practical_cv\Lights\model\yolo11s\train5\weights\best.pt")

    validation_results = model.val(data=r'D:\MyFile\qq_3045834499\yolo11-chest\ultralytics\cfg\datasets\A_my_data.yaml',
                                   imgsz=512, batch=16, device="0", workers=0)
