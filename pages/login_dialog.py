from PyQt5.QtWidgets import (QDialog, QLineEdit, QPushButton, QVBoxLayout,
                             QFormLayout, QHBoxLayout, QLabel, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush
from PyQt5.QtCore import Qt, pyqtSignal
from pages.register_dialog import RegisterDialog
from core.database_handler import check_user
import os

class LoginDialog(QDialog):
    login_success = pyqtSignal(str)  # 登录成功信号，传递用户名

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("掌静脉识别系统 - 用户登录")
        self.resize(600, 400)
        self.setup_ui()

    def setup_ui(self):
        # 设置背景 - 修正版
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 构建图片路径
            bg_path = os.path.join(current_dir, "..", "images", "ui", "bgd.png")

            # 检查图片是否存在
            if not os.path.exists(bg_path):
                raise FileNotFoundError(f"背景图片不存在: {bg_path}")

            # 加载图片
            pixmap = QPixmap(bg_path)
            if pixmap.isNull():
                raise Exception("图片加载失败，可能是文件损坏或格式不支持")

            # 设置调色板
            palette = self.palette()
            palette.setBrush(QPalette.Window,
                             QBrush(pixmap.scaled(
                                 self.size(),
                                 Qt.IgnoreAspectRatio,
                                 Qt.SmoothTransformation)))
            self.setPalette(palette)
            self.setAutoFillBackground(True)  # 必须调用这个！

        except Exception as e:
            print(f"设置背景失败: {str(e)}")
            # 设置备用背景色
            palette = self.palette()
            palette.setColor(QPalette.Window, Qt.lightGray)  # 浅灰色作为备用
            self.setPalette(palette)
            self.setAutoFillBackground(True)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)

        # 标题
        title_label = QLabel("用户登录")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
            }
        """)

        # 表单
        form_layout = QFormLayout()
        form_layout.setSpacing(20)

        self.username = QLineEdit()
        self.username.setPlaceholderText("请输入用户名")
        self.password = QLineEdit()
        self.password.setPlaceholderText("请输入密码")
        self.password.setEchoMode(QLineEdit.Password)

        # 设置输入框样式
        input_style = """
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.8);
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """
        self.username.setStyleSheet(input_style)
        self.password.setStyleSheet(input_style)

        form_layout.addRow("用户名:", self.username)
        form_layout.addRow("密码:", self.password)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.login_btn = QPushButton("登录")
        self.register_btn = QPushButton("注册")
        self.guest_btn = QPushButton("游客访问")

        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton#guest_btn {
                background-color: #95a5a6;
            }
            QPushButton#guest_btn:hover {
                background-color: #7f8c8d;
            }
        """
        self.login_btn.setStyleSheet(button_style)
        self.register_btn.setStyleSheet(button_style)
        self.guest_btn.setObjectName("guest_btn")
        self.guest_btn.setStyleSheet(button_style)

        button_layout.addWidget(self.login_btn)
        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.guest_btn)

        # 添加到主布局
        main_layout.addWidget(title_label)
        main_layout.addSpacing(30)
        main_layout.addLayout(form_layout)
        main_layout.addSpacing(30)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # 连接信号
        self.login_btn.clicked.connect(self.on_login)
        self.register_btn.clicked.connect(self.show_register)
        self.guest_btn.clicked.connect(self.on_guest)

    def on_login(self):
        username = self.username.text()
        password = self.password.text()

        if not username or not password:
            QMessageBox.warning(self, "提示", "用户名和密码不能为空!")
            return

        if check_user(username, password):
            self.login_success.emit(username)  # 发射登录成功信号
            self.accept()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误!")

    def show_register(self):
        register_dialog = RegisterDialog(self)
        if register_dialog.exec_() == QDialog.Accepted:
            self.username.setText(register_dialog.username.text())
            self.password.setText(register_dialog.password.text())

    def on_guest(self):
        self.reject()