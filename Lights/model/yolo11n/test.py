from ultralytics import YOLO

# 加载模型
model = YOLO(r'D:\MyFile\qq_3045834499\yolo11-chest\42_demo\train3\weights\best.pt')

# 指定要检测的图片路径
img_path = r'D:\Programming\Pycharm\Comprehensive_Practical_cv\Lights\test\data\output_image.jpg'

# 指定多个图像路径
img_paths = [
    r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\images\000038.jpg',
    r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\images\000066.jpg',
    r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\images\000046.jpg',
    r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\images\000056.jpg',
]

# 进行检测
results = model.predict(source=img_paths)

# 显示检测结果
for r in results:
    im_array = r.plot()  # 绘制检测结果
    import cv2
    cv2.imshow('YOLOv8 Inference', im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()