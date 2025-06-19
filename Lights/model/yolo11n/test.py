from ultralytics import YOLO

# 加载模型
model = YOLO(r'D:\MyFile\qq_3045834499\yolo11-chest\42_demo\train3\weights\best.pt')

# 指定要检测的图片路径
img_path = r'D:\Programming\Pycharm\Comprehensive_Practical_cv\Lights\test\data\output_image.jpg'

# 进行检测
results = model.predict(source=img_path)

# 显示检测结果
for r in results:
    im_array = r.plot()  # 绘制检测结果
    import cv2
    cv2.imshow('YOLOv8 Inference', im_array)
    cv2.waitKey(0)