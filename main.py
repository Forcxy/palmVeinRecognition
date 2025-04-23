import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QDialog
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from PyQt5.QtCore import Qt
from pages.login_dialog import LoginDialog
from pages.main_menu import MainMenu
from pages.vein_recognition import VeinRecognitionWindow
from core.database_handler import create_user_db, clear_feature_database, check_user, is_admin, update_last_login, init_databases
from pages.batchROI import BatchROIWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("掌静脉识别系统")
        self.resize(1200, 800)
        self.is_admin = False  # 当前用户是否是管理员

        # 创建用户数据库
        create_user_db()

        # 显示登录对话框
        self.show_login()

        # 创建堆叠窗口
        self.stacked_widget = QStackedWidget()

        # 添加主菜单和掌静脉识别界面
        self.main_menu = MainMenu(self)
        self.vein_recognition_window = VeinRecognitionWindow(self)
        self.batch_roi_window = BatchROIWindow(self)  # 新增

        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.vein_recognition_window)
        self.stacked_widget.addWidget(self.batch_roi_window)  # 新增

        self.setCentralWidget(self.stacked_widget)

        # 设置背景
        self.set_background()

    def set_background(self):
        palette = QPalette()
        bg_path = os.path.join(os.path.dirname(__file__), "images", "bgd.png")
        if os.path.exists(bg_path):
            palette.setBrush(
                QPalette.Window,
                QBrush(QPixmap(bg_path).scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            )
            self.setAutoFillBackground(True)
            self.setPalette(palette)

    def show_login(self):
        login_dialog = LoginDialog(self)
        result = login_dialog.exec_()

        if result == QDialog.Accepted:
            username = login_dialog.username.text()
            self.is_admin = is_admin(username)
            update_last_login(username)
            QMessageBox.information(self, "登录成功", f"欢迎, {username}!")
        else:
            if result == QDialog.Rejected:
                QMessageBox.information(self, "游客模式", "您将以游客身份访问系统")


if __name__ == "__main__":
    # 1. 首先初始化数据库（确保表存在）
    init_databases()
    # 2. 然后才清空数据（如果需要）
    clear_feature_database()  # 注意：生产环境应移除这行
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())