import sys
from PyQt5.QtWidgets import QApplication, QMainWindow  # PyQt5 导入
# 或者使用 PySide6:
# from PySide6.QtWidgets import QApplication, QMainWindow

# 导入生成的 UI 类（假设类名为 Ui_MainWindow）
from MainWindow import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建 UI 实例并设置到当前窗口
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # 关键步骤：将UI附加到当前窗口


if __name__ == "__main__":
    # 创建应用实例
    app = QApplication(sys.argv)

    # 创建主窗口并显示
    window = MainWindow()
    window.show()

    # 启动事件循环
    sys.exit(app.exec_())