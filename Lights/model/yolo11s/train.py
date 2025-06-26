import os
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\Programming\Pycharm\Comprehensive_Practical_cv\Lights\model\yolo11s\train4\weights\last.pt")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    results = model.train(
        data=r'D:\MyFile\qq_3045834499\yolo11-chest\ultralytics\cfg\datasets\A_my_data.yaml',
        project="./", epochs=80, imgsz=512, device=[0], workers=0, batch=16, cache=True,
        save = True, save_period = 10
    )
