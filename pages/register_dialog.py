from PyQt5.QtWidgets import (QDialog, QLineEdit, QPushButton, QVBoxLayout,
                             QFormLayout, QHBoxLayout, QLabel, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush
from PyQt5.QtCore import Qt
from core.database_handler import add_user
import os

class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("掌静脉识别系统 - 用户注册")
        self.resize(600, 450)
        self.setup_ui()

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
            self.setAutoFillBackground(True)  # 必须调用这个

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
        title_label = QLabel("用户注册")
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
        self.username.setPlaceholderText("4-16位字母或数字")
        self.password = QLineEdit()
        self.password.setPlaceholderText("至少6位字符")
        self.password.setEchoMode(QLineEdit.Password)
        self.confirm_password = QLineEdit()
        self.confirm_password.setPlaceholderText("再次输入密码")
        self.confirm_password.setEchoMode(QLineEdit.Password)

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
        for input_field in [self.username, self.password, self.confirm_password]:
            input_field.setStyleSheet(input_style)

        form_layout.addRow("用户名:", self.username)
        form_layout.addRow("密码:", self.password)
        form_layout.addRow("确认密码:", self.confirm_password)

        # 按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.register_btn = QPushButton("注册")
        self.cancel_btn = QPushButton("取消")

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
            QPushButton#cancel_btn {
                background-color: #e74c3c;
            }
            QPushButton#cancel_btn:hover {
                background-color: #c0392b;
            }
        """
        self.register_btn.setStyleSheet(button_style)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setStyleSheet(button_style)

        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.cancel_btn)

        # 添加到主布局
        main_layout.addWidget(title_label)
        main_layout.addSpacing(30)
        main_layout.addLayout(form_layout)
        main_layout.addSpacing(30)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # 连接信号
        self.register_btn.clicked.connect(self.on_register)
        self.cancel_btn.clicked.connect(self.reject)

    def on_register(self):
        username = self.username.text().strip()
        password = self.password.text().strip()
        confirm_password = self.confirm_password.text().strip()

        if not username or not password or not confirm_password:
            QMessageBox.warning(self, "提示", "所有字段都必须填写!")
            return

        if len(username) < 4 or len(username) > 16:
            QMessageBox.warning(self, "错误", "用户名长度必须在4-16个字符之间!")
            return

        if not username.isalnum():
            QMessageBox.warning(self, "错误", "用户名只能包含字母和数字!")
            return

        if len(password) < 6:
            QMessageBox.warning(self, "错误", "密码长度至少需要6个字符!")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致!")
            return

        if add_user(username, password):
            QMessageBox.information(self, "成功", "注册成功!")
            self.accept()
        else:
            QMessageBox.warning(self, "错误", "用户名已存在!")