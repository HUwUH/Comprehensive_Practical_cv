# -*- coding: utf-8 -*-
import sys
import os

import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QHBoxLayout, QPushButton
from MainWindow import Ui_MainWindow  # 替换为你的UI文件名
from ultralytics import YOLO

class ZoomDialog(QDialog):
    """自定义放大对话框，用于显示放大的图像"""

    def __init__(self, title, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"放大视图: {title}")
        self.setMinimumSize(800, 600)

        # 创建主布局
        main_layout = QVBoxLayout()

        # 创建图形画布
        self.canvas = FigureCanvas(fig)
        main_layout.addWidget(self.canvas)

        # 创建关闭按钮
        btn_layout = QHBoxLayout()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        btn_layout.addStretch()

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)


class MainApplication(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化状态变量
        self.current_file = None
        self.current_folder = None
        self.current_numpy_data = None
        self.compare_numpy_data = None
        self.viewpoint = [0, 0, 0]  # [x, y, z]
        self.file_mode = 0  # 0:未选择, 1:文件, 2:文件夹
        self.current_dcm_index = 0
        self.dcm_files = []
        self.dcm_positions = {}  # 存储DICOM文件的位置信息
        self.three_d_model = None  # 存储3D模型数据

        # 存储每个视图的图形对象
        self.view_figures = {
            'view1': None,
            'view2': None,
            'view3': None,
            'view4': None
        }

        # 连接按钮信号
        self.pushButton_2.clicked.connect(self.select_folder)
        self.pushButton_3.clicked.connect(self.select_file)
        self.pushButton.clicked.connect(self.send_address)
        self.pushButton_4.clicked.connect(self.open_numpy_file)
        self.pushButton_5.clicked.connect(self.predict1)
        self.pushButton_6.clicked.connect(self.accuracy_analysis)
        self.pushButton_7.clicked.connect(self.exit_app)

        # 初始化图像视图
        self.init_graphics_views()

        # 设置鼠标滚轮事件
        self.graphicsView.wheelEvent = self.view1_wheel_event
        self.graphicsView_2.wheelEvent = self.view2_wheel_event
        self.graphicsView_3.wheelEvent = self.view3_wheel_event
        self.graphicsView_4.wheelEvent = self.view4_wheel_event

        # 设置鼠标双击事件
        self.graphicsView.mouseDoubleClickEvent = lambda event: self.handle_double_click(event, 'view1')
        self.graphicsView_2.mouseDoubleClickEvent = lambda event: self.handle_double_click(event, 'view2')
        self.graphicsView_3.mouseDoubleClickEvent = lambda event: self.handle_double_click(event, 'view3')
        self.graphicsView_4.mouseDoubleClickEvent = lambda event: self.handle_double_click(event, 'view4')

        # 设置当前激活的视图
        self.active_view = 1

    def init_graphics_views(self):
        """初始化四个图形视图"""
        self.clear_all_views()

        # 为每个视图创建图形画布
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)
        self.scene1 = QtWidgets.QGraphicsScene()
        self.scene1.addWidget(self.canvas1)
        self.graphicsView.setScene(self.scene1)
        self.view_figures['view1'] = self.fig1

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        self.scene2 = QtWidgets.QGraphicsScene()
        self.scene2.addWidget(self.canvas2)
        self.graphicsView_2.setScene(self.scene2)
        self.view_figures['view2'] = self.fig2

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvas(self.fig3)
        self.scene3 = QtWidgets.QGraphicsScene()
        self.scene3.addWidget(self.canvas3)
        self.graphicsView_3.setScene(self.scene3)
        self.view_figures['view3'] = self.fig3

        # 视图4使用3D轴
        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111, projection='3d')
        self.canvas4 = FigureCanvas(self.fig4)
        self.scene4 = QtWidgets.QGraphicsScene()
        self.scene4.addWidget(self.canvas4)
        self.graphicsView_4.setScene(self.scene4)
        self.view_figures['view4'] = self.fig4

        # 显示占位文本
        self.ax1.text(0.5, 0.5, "视图1\n(主视图)",
                      ha='center', va='center', fontsize=20, color='gray')
        self.ax2.text(0.5, 0.5, "视图2\n(俯视图)",
                      ha='center', va='center', fontsize=20, color='gray')
        self.ax3.text(0.5, 0.5, "视图3\n(侧视图)",
                      ha='center', va='center', fontsize=20, color='gray')

        # 视图4显示3D占位文本
        self.ax4.text2D(0.5, 0.5, "视图4\n(3D模型)",
                        ha='center', va='center', fontsize=20, color='gray')
        self.ax4.set_axis_off()

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

    def clear_all_views(self):
        """清除所有视图的内容"""
        self.graphicsView.setScene(QtWidgets.QGraphicsScene())
        self.graphicsView_2.setScene(QtWidgets.QGraphicsScene())
        self.graphicsView_3.setScene(QtWidgets.QGraphicsScene())
        self.graphicsView_4.setScene(QtWidgets.QGraphicsScene())

    def handle_double_click(self, event, view_name):
        """处理鼠标双击事件 - 在新窗口中显示放大视图"""
        if event.button() != QtCore.Qt.LeftButton:
            return

        # 获取当前视图的图形对象
        fig = self.view_figures[view_name]

        # 创建新的图形对象用于放大视图
        zoom_fig = Figure(figsize=(10, 8))
        zoom_ax = zoom_fig.add_subplot(111)

        # 复制原始图形的内容
        if view_name == 'view1':
            for line in self.ax1.get_lines():
                zoom_ax.add_line(line)
            if self.ax1.images:
                img = self.ax1.images[0]
                zoom_ax.imshow(img.get_array(), cmap=img.get_cmap(),
                               vmin=img.get_clim()[0], vmax=img.get_clim()[1])
            zoom_ax.set_title(self.ax1.get_title())
            zoom_ax.axis('off')
        elif view_name == 'view2':
            for line in self.ax2.get_lines():
                zoom_ax.add_line(line)
            if self.ax2.images:
                img = self.ax2.images[0]
                zoom_ax.imshow(img.get_array(), cmap=img.get_cmap(),
                               vmin=img.get_clim()[0], vmax=img.get_clim()[1])
            zoom_ax.set_title(self.ax2.get_title())
            zoom_ax.axis('off')
        elif view_name == 'view3':
            for line in self.ax3.get_lines():
                zoom_ax.add_line(line)
            if self.ax3.images:
                img = self.ax3.images[0]
                zoom_ax.imshow(img.get_array(), cmap=img.get_cmap(),
                               vmin=img.get_clim()[0], vmax=img.get_clim()[1])
            zoom_ax.set_title(self.ax3.get_title())
            zoom_ax.axis('off')
        elif view_name == 'view4':
            # 对于视图4，特殊处理（可能是3D图）
            if hasattr(self.ax4, 'name') and self.ax4.name == '3d':
                # 3D视图 - 创建新的3D视图
                zoom_fig.clf()
                zoom_ax = zoom_fig.add_subplot(111, projection='3d')

                # 复制3D内容
                for collection in self.ax4.collections:
                    zoom_ax.add_collection3d(collection)

                # 复制轴设置
                zoom_ax.set_xlim(self.ax4.get_xlim())
                zoom_ax.set_ylim(self.ax4.get_ylim())
                zoom_ax.set_zlim(self.ax4.get_zlim())
                zoom_ax.set_xlabel(self.ax4.get_xlabel())
                zoom_ax.set_ylabel(self.ax4.get_ylabel())
                zoom_ax.set_zlabel(self.ax4.get_zlabel())
                zoom_ax.set_title(self.ax4.get_title())

                # 复制视角
                zoom_ax.view_init(elev=self.ax4.elev, azim=self.ax4.azim)
            else:
                # 2D视图
                for line in self.ax4.get_lines():
                    zoom_ax.add_line(line)
                if self.ax4.images:
                    img = self.ax4.images[0]
                    zoom_ax.imshow(img.get_array(), cmap=img.get_cmap(),
                                   vmin=img.get_clim()[0], vmax=img.get_clim()[1])
                zoom_ax.set_title(self.ax4.get_title())
                if self.ax4.axis() != 'off':
                    zoom_ax.axis('on')

        # 创建并显示放大对话框
        dialog = ZoomDialog(view_name, zoom_fig, self)
        dialog.exec_()

    # ===================== 按钮功能实现 =====================

    def select_folder(self):
        """选择文件夹并加载DICOM文件"""
        folder = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if folder:
            self.current_folder = folder
            self.file_mode = 2  # 文件夹模式

            # 获取文件夹中的所有DICOM文件
            all_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith('.dcm')
            ]

            if not all_files:
                QMessageBox.warning(self, "无DICOM文件", "选择的文件夹中没有找到DICOM文件!")
                return

            # 读取文件位置信息并排序
            self.dcm_positions = {}
            self.dcm_files = []

            # 创建进度对话框
            progress = QtWidgets.QProgressDialog("加载DICOM文件...", "取消", 0, len(all_files), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("加载中")

            for i, file_path in enumerate(all_files):
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()

                if progress.wasCanceled():
                    break

                try:
                    # 读取DICOM文件元数据
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                    # 获取位置信息
                    if hasattr(ds, 'ImagePositionPatient'):
                        position = list(map(float, ds.ImagePositionPatient))
                        # 使用y坐标作为排序依据
                        y_position = position[1] if len(position) > 1 else 0.0
                    else:
                        y_position = 0.0

                    # 存储位置信息
                    self.dcm_positions[file_path] = y_position

                except Exception as e:
                    print(f"读取DICOM文件元数据失败: {file_path} - {str(e)}")

            progress.setValue(len(all_files))

            # 按y位置排序文件
            self.dcm_files = sorted(all_files, key=lambda f: self.dcm_positions.get(f, 0.0))

            if not self.dcm_files:
                QMessageBox.warning(self, "无有效文件", "未能加载任何有效的DICOM文件!")
                return

            self.current_dcm_index = 0
            self.load_and_display_dcm(self.dcm_files[self.current_dcm_index])

            # 更新视图4显示位置信息
            self.update_position_display()

            self.statusbar.showMessage(f"已加载文件夹: {folder}, 共{len(self.dcm_files)}个DICOM文件 (按Y位置排序)",
                                       5000)

    def select_file(self):
        """选择单个DICOM文件"""
        file, _ = QFileDialog.getOpenFileName(
            self, "选择DICOM文件", "", "DICOM Files (*.dcm)"
        )
        if file:
            self.current_file = file
            self.file_mode = 1  # 文件模式
            self.dcm_files = [file]  # 为了统一处理，也放入列表
            self.current_dcm_index = 0
            self.load_and_display_dcm(file)

            # 尝试读取位置信息
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    position = list(map(float, ds.ImagePositionPatient))
                    self.dcm_positions[file] = position[1] if len(position) > 1 else 0.0
                else:
                    self.dcm_positions[file] = 0.0
            except Exception:
                self.dcm_positions[file] = 0.0

            # 更新视图4显示位置信息
            self.update_position_display()

            self.statusbar.showMessage(f"已加载文件: {file}", 5000)

    def update_position_display(self):
        """在视图4中显示当前DICOM文件的位置信息"""
        if not self.dcm_files:
            return

        current_file = self.dcm_files[self.current_dcm_index]
        y_position = self.dcm_positions.get(current_file, 0.0)

        # 清除视图4并创建2D轴
        self.fig4.clf()
        self.ax4 = self.fig4.add_subplot(111)

        info = f"当前DICOM文件: {os.path.basename(current_file)}\n"
        info += f"位置索引: {self.current_dcm_index + 1}/{len(self.dcm_files)}\n"
        info += f"Y位置: {y_position:.2f}\n"

        # 显示位置分布图
        if len(self.dcm_files) > 1:
            positions = [self.dcm_positions.get(f, 0.0) for f in self.dcm_files]
            self.ax4.plot(positions, 'b-', label='Y位置')
            self.ax4.plot(self.current_dcm_index, y_position, 'ro', markersize=8, label='当前')
            self.ax4.set_xlabel('文件索引')
            self.ax4.set_ylabel('Y位置')
            self.ax4.set_title('DICOM文件位置分布')
            self.ax4.legend()
            self.ax4.grid(True)
        else:
            self.ax4.text(0.5, 0.5, info,
                          ha='center', va='center', fontsize=12)
            self.ax4.axis('off')

        self.canvas4.draw()

    def load_and_display_dcm(self, file_path):
        """加载并显示DICOM文件到视图1"""
        try:
            # 读取DICOM文件
            ds = pydicom.dcmread(file_path)

            # 获取像素数据
            pixel_array = ds.pixel_array

            # 清除视图1并显示新图像
            self.ax3.clear()
            self.ax3.imshow(pixel_array, cmap='gray')

            # 显示文件名和位置信息
            y_position = self.dcm_positions.get(file_path, 0.0)
            title = f"DICOM图像: {os.path.basename(file_path)}"
            if len(self.dcm_files) > 1:
                title += f"\n位置: {self.current_dcm_index + 1}/{len(self.dcm_files)} (Y={y_position:.2f})"

            self.ax3.set_title(title)
            self.ax3.axis('off')
            self.canvas3.draw()

            # 在状态栏显示DICOM信息
            info = f"图像尺寸: {pixel_array.shape} | 患者: {ds.get('PatientName', '未知')}"
            self.statusbar.showMessage(info, 5000)

        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"无法加载DICOM文件:\n{str(e)}")

    def send_address(self):
        """发送当前选择的文件或文件夹地址"""
        if self.file_mode == 1 and self.current_file:
            # 文件模式

            #self.load_and_display_dcm(results[0])
            QMessageBox.information(self, "发送地址", f"已发送文件地址到后端:\n{self.current_file}")
            # 这里可以添加实际的后端通信代码
            print(f"发送文件地址: {self.current_file}")

        elif self.file_mode == 2 and self.current_folder:
            # 文件夹模式
            QMessageBox.information(self, "发送地址", f"已发送文件夹地址到后端:\n{self.current_folder}")
            # 这里可以添加实际的后端通信代码
            print(f"发送文件夹地址: {self.current_folder}")

        else:
            QMessageBox.warning(self, "无选择", "请先选择文件或文件夹!")

    def open_numpy_file(self):
        """打开并分析numpy文件，创建3D模型"""
        file, _ = QFileDialog.getOpenFileName(
            self, "打开Numpy文件", "", "Numpy Files (*.npy *.npz)"
        )
        if file:
            try:
                # 加载numpy数据
                self.current_numpy_data = np.load(file)

                # 检查数据维度
                if len(self.current_numpy_data.shape) != 3:
                    QMessageBox.warning(self, "数据格式错误", "需要3D numpy数组!")
                    return

                # 初始化视角点为中心位置
                self.viewpoint = [
                    self.current_numpy_data.shape[0] // 2,
                    self.current_numpy_data.shape[1] // 2,
                    self.current_numpy_data.shape[2] // 2
                ]

                # 更新三个视图
                self.update_three_views()

                # 创建3D模型
                self.create_3d_model()

                self.statusbar.showMessage(f"已加载numpy文件: {file}", 5000)

            except Exception as e:
                QMessageBox.critical(self, "加载错误", f"无法加载numpy文件:\n{str(e)}")

    def create_3d_model(self):
        """创建3D模型并显示在视图4中"""
        if self.current_numpy_data is None:
            return

        # 创建进度对话框
        progress = QProgressDialog("创建3D模型...", "取消", 0, 100, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("3D重建")
        progress.setValue(10)
        QtWidgets.QApplication.processEvents()

        try:
            # 使用Marching Cubes算法提取等值面
            # 为了性能，我们可以对数据进行降采样
            downsampling_factor = 2 if max(self.current_numpy_data.shape) > 256 else 1

            # 准备降采样后的数据
            data = self.current_numpy_data[::downsampling_factor,
                   ::downsampling_factor,
                   ::downsampling_factor]

            progress.setValue(30)
            QtWidgets.QApplication.processEvents()

            # 提取等值面 - 使用中值作为阈值
            threshold = np.median(data)
            verts, faces, _, _ = measure.marching_cubes(data, level=threshold)

            progress.setValue(70)
            QtWidgets.QApplication.processEvents()

            # 创建3D网格
            mesh = Poly3DCollection(verts[faces], alpha=0.3)
            mesh.set_edgecolor('k')
            mesh.set_facecolor([0.45, 0.45, 0.75])  # 设置表面颜色

            # 清除视图4并创建3D轴
            self.fig4.clf()
            self.ax4 = self.fig4.add_subplot(111, projection='3d')

            # 添加网格到轴
            self.ax4.add_collection3d(mesh)

            # 设置轴限制
            self.ax4.set_xlim(0, data.shape[0])
            self.ax4.set_ylim(0, data.shape[1])
            self.ax4.set_zlim(0, data.shape[2])

            # 设置轴标签
            self.ax4.set_xlabel('X')
            self.ax4.set_ylabel('Y')
            self.ax4.set_zlabel('Z')
            self.ax4.set_title('3D重建模型')

            # 设置初始视角
            self.ax4.view_init(elev=30, azim=45)

            progress.setValue(90)
            QtWidgets.QApplication.processEvents()

            self.canvas4.draw()

            progress.setValue(100)
            self.statusbar.showMessage("3D模型创建完成! 使用鼠标拖拽旋转视角", 5000)

            # 存储模型数据
            self.three_d_model = {
                'verts': verts,
                'faces': faces,
                'downsampling_factor': downsampling_factor
            }

        except Exception as e:
            QMessageBox.critical(self, "3D重建错误", f"创建3D模型时出错:\n{str(e)}")
            # 回退到显示位置信息
            self.update_position_display()

    def update_three_views(self):
        """根据当前视角点更新三个视图"""
        if self.current_numpy_data is None:
            return

        # 获取当前视角点
        x, y, z = self.viewpoint

        # 视图1: xy平面 (z固定)
        self.ax1.clear()
        self.ax1.imshow(self.current_numpy_data[:, :, z], cmap='gray')
        self.ax1.set_title(f"XY平面 (Z={z})")
        self.ax1.axis('off')
        self.canvas1.draw()

        # 视图2: xz平面 (y固定)
        self.ax2.clear()
        self.ax2.imshow(self.current_numpy_data[:, y, :], cmap='gray')
        self.ax2.set_title(f"XZ平面 (Y={y})")
        self.ax2.axis('off')
        self.canvas2.draw()

        # 视图3: yz平面 (x固定)
        self.ax3.clear()
        self.ax3.imshow(self.current_numpy_data[x, :, :], cmap='gray')
        self.ax3.set_title(f"YZ平面 (X={x})")
        self.ax3.axis('off')
        self.canvas3.draw()

    def predict1(self):
        """根据当前模式进行预测"""
        if self.file_mode == 0:
            QMessageBox.warning(self, "未选择", "请先选择文件或文件夹!")
            return

        # 根据模式调用不同的预测函数
        if self.file_mode == 1:
            # 文件模式预测
            self.predict_file()
        elif self.file_mode == 2:
            # 文件夹模式预测
            self.predict_folder()

    def predict_file(self):
        """对单个文件进行预测"""
        if not self.current_file:
            QMessageBox.warning(self, "无文件", "未选择文件!")
            return
        model = YOLO(r'D:\PycharmProjects\Comprehensive_Practical_cv\model\best.pt')
        img_path = r"D:\PycharmProjects\Comprehensive_Practical_cv\jpgfile\temp.jpg"
        self.dcm_to_jpg(self.current_file, img_path)
        results = model.predict(source=img_path)
        result = results[0]

        #self.ax1.clear()
        #self.ax1.imshow(results[0], cmap='viridis')
        # 这里添加实际的预测逻辑
        # 示例: 显示预测结果
        self.ax2.clear()
        #self.ax1.text(0.5, 0.5, "预测结果", ha='center', va='center', fontsize=20)
        self.ax2.imshow(result.plot(), cmap='viridis')
        self.ax2.axis('off')
        self.canvas2.draw()

        self.statusbar.showMessage(f"已完成对 {os.path.basename(self.current_file)} 的预测", 5000)

    def predict_folder(self):
        """对整个文件夹进行预测"""
        if not self.current_folder or not self.dcm_files:
            QMessageBox.warning(self, "无文件夹", "未选择有效文件夹!")
            return

        # 这里添加实际的预测逻辑
        # 示例: 显示预测进度
        self.ax1.clear()
        self.ax1.text(0.5, 0.5, f"预测中...\n0/{len(self.dcm_files)}",
                      ha='center', va='center', fontsize=16)
        self.ax1.axis('off')
        self.canvas1.draw()

        # 创建进度对话框
        progress = QtWidgets.QProgressDialog("预测处理中...", "取消", 0, len(self.dcm_files), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("预测进度")

        # 模拟预测过程
        for i in range(len(self.dcm_files)):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            QtWidgets.QApplication.processEvents()

            # 更新进度
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, f"预测中...\n{i + 1}/{len(self.dcm_files)}",
                          ha='center', va='center', fontsize=16)
            self.ax1.axis('off')
            self.canvas1.draw()

            # 模拟处理时间
            QtCore.QThread.msleep(50)

        progress.setValue(len(self.dcm_files))

        self.statusbar.showMessage(f"已完成对文件夹 {os.path.basename(self.current_folder)} 的预测", 5000)

    def accuracy_analysis(self):
        """进行准确度分析"""
        if self.current_numpy_data is None:
            QMessageBox.warning(self, "无数据", "请先打开numpy文件!")
            return

        # 选择用于比较的numpy文件
        file, _ = QFileDialog.getOpenFileName(
            self, "选择对比Numpy文件", "", "Numpy Files (*.npy *.npz)"
        )
        if file:
            try:
                # 加载对比数据
                self.compare_numpy_data = np.load(file)

                # 检查数据维度是否匹配
                if self.current_numpy_data.shape != self.compare_numpy_data.shape:
                    QMessageBox.warning(self, "维度不匹配", "两个numpy数组的维度不一致!")
                    return

                # 计算准确度指标
                accuracy = self.calculate_accuracy()

                # 显示分析结果
                QMessageBox.information(
                    self,
                    "准确度分析结果",
                    f"准确率: {accuracy['accuracy']:.2%}\n"
                    f"召回率: {accuracy['recall']:.2%}\n"
                    f"精确率: {accuracy['precision']:.2%}\n"
                    f"F1分数: {accuracy['f1_score']:.2f}"
                )

                # 在视图4中显示混淆矩阵
                self.display_confusion_matrix(accuracy['confusion_matrix'])

            except Exception as e:
                QMessageBox.critical(self, "加载错误", f"无法加载对比numpy文件:\n{str(e)}")

    def calculate_accuracy(self):
        """计算准确度指标（简化版）"""
        # 这里使用简化计算，实际应用中应根据具体问题实现
        diff = np.abs(self.current_numpy_data - self.compare_numpy_data)

        # 假设二分类问题
        true_positives = np.sum((self.current_numpy_data > 0.5) & (self.compare_numpy_data > 0.5))
        true_negatives = np.sum((self.current_numpy_data <= 0.5) & (self.compare_numpy_data <= 0.5))
        false_positives = np.sum((self.current_numpy_data > 0.5) & (self.compare_numpy_data <= 0.5))
        false_negatives = np.sum((self.current_numpy_data <= 0.5) & (self.compare_numpy_data > 0.5))

        # 计算指标
        accuracy = (true_positives + true_negatives) / self.current_numpy_data.size
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 创建混淆矩阵
        confusion_matrix = np.array([
            [true_negatives, false_positives],
            [false_negatives, true_positives]
        ])

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix
        }

    def display_confusion_matrix(self, matrix):
        """在视图4中显示混淆矩阵"""
        # 清除视图4并创建2D轴
        self.fig4.clf()
        self.ax4 = self.fig4.add_subplot(111)

        # 绘制混淆矩阵
        cax = self.ax4.matshow(matrix, cmap='Blues')
        self.fig4.colorbar(cax)

        # 添加数值标签
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.ax4.text(j, i, str(matrix[i, j]),
                              ha='center', va='center',
                              color='black' if matrix[i, j] < np.max(matrix) / 2 else 'white')

        # 设置坐标轴标签
        self.ax4.set_xticks([0, 1])
        self.ax4.set_yticks([0, 1])
        self.ax4.set_xticklabels(['预测负', '预测正'])
        self.ax4.set_yticklabels(['实际负', '实际正'])
        self.ax4.set_title('混淆矩阵')

        self.canvas4.draw()

    def exit_app(self):
        """退出应用程序"""
        QtWidgets.QApplication.quit()

    # ===================== 滚轮事件处理 =====================

    def view1_wheel_event(self, event):
        """视图1的滚轮事件"""
        if self.current_numpy_data is not None:
            # numpy模式：调整Z值
            self.active_view = 1
            delta = event.angleDelta().y() // 120  # 获取滚轮滚动方向
            new_z = self.viewpoint[2] + delta

            # 确保在有效范围内
            if 0 <= new_z < self.current_numpy_data.shape[2]:
                self.viewpoint[2] = new_z
                self.update_three_views()
        elif self.file_mode == 2 and self.dcm_files:
            # DICOM文件夹模式：切换DICOM图像
            delta = event.angleDelta().y() // 120
            new_index = self.current_dcm_index + delta

            # 确保索引在有效范围内
            if 0 <= new_index < len(self.dcm_files):
                self.current_dcm_index = new_index
                self.load_and_display_dcm(self.dcm_files[self.current_dcm_index])
                self.update_position_display()

    def view2_wheel_event(self, event):
        """视图2的滚轮事件 - XZ平面 (调整Y值)"""
        if self.current_numpy_data is not None:
            self.active_view = 2
            delta = event.angleDelta().y() // 120
            new_y = self.viewpoint[1] + delta

            if 0 <= new_y < self.current_numpy_data.shape[1]:
                self.viewpoint[1] = new_y
                self.update_three_views()

    def view3_wheel_event(self, event):
        """视图3的滚轮事件 - YZ平面 (调整X值)"""
        if self.current_numpy_data is not None:
            self.active_view = 3
            delta = event.angleDelta().y() // 120
            new_x = self.viewpoint[0] + delta

            if 0 <= new_x < self.current_numpy_data.shape[0]:
                self.viewpoint[0] = new_x
                self.update_three_views()

    def view4_wheel_event(self, event):
        """视图4的滚轮事件 - 3D视图 (缩放)"""
        if self.three_d_model is not None:
            # 获取当前缩放比例
            current_scale = self.ax4.get_scale()

            # 根据滚轮方向调整缩放比例
            delta = event.angleDelta().y() // 120
            new_scale = current_scale * (1.0 + delta * 0.1)

            # 设置缩放限制
            new_scale = max(0.5, min(new_scale, 5.0))

            # 重新绘制视图
            self.ax4.set_box_aspect([1, 1, 1])
            self.canvas4.draw()

    def dcm_to_jpg(self, dcm_path, jpg_path, window_center=-600, window_width=1600):
        # 读取 DICOM 文件
        dcm = pydicom.dcmread(dcm_path)
        # 获取像素数据
        pixel_data = dcm.pixel_array.astype(np.float32)
        # 获取 DICOM 文件中的斜率和截距
        slope = dcm.RescaleSlope
        intercept = dcm.RescaleIntercept
        # 应用斜率和截距进行线性变换
        hu = pixel_data * slope + intercept

        # 计算窗位和窗宽的上下限
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        # 进行窗宽窗位的调整
        windowed = np.clip(hu, window_min, window_max)
        # 归一化到 0-255 范围
        windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)

        # 保存为 JPEG 图片
        cv2.imwrite(jpg_path, windowed)



# ===================== 主函数 =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用现代样式

    # 设置应用样式
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197).lighter())
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    main_window = MainApplication()
    main_window.setWindowTitle("医学影像分析系统")
    main_window.show()
    sys.exit(app.exec_())