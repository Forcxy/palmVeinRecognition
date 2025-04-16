from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMessageBox
from PyQt5.QtGui import QFont


class MainMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        # 创建菜单按钮
        self.vein_recognition_btn = QPushButton("掌静脉识别")
        self.user_management_btn = QPushButton("用户管理")
        self.system_settings_btn = QPushButton("系统设置")
        self.exit_btn = QPushButton("退出系统")

        # 设置按钮样式
        font = QFont("华文中宋", 14)
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#admin_btn {
                background-color: #3498db;
            }
            QPushButton#admin_btn:hover {
                background-color: #2980b9;
            }
        """

        for btn in [self.vein_recognition_btn, self.system_settings_btn, self.exit_btn]:
            btn.setFont(font)
            btn.setStyleSheet(button_style)

        # 用户管理按钮特殊样式
        self.user_management_btn.setFont(font)
        self.user_management_btn.setObjectName("admin_btn")
        self.user_management_btn.setStyleSheet(button_style)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.vein_recognition_btn)
        layout.addWidget(self.user_management_btn)
        layout.addWidget(self.system_settings_btn)
        layout.addWidget(self.exit_btn)
        layout.addStretch(1)

        self.setLayout(layout)

        # 连接信号
        self.vein_recognition_btn.clicked.connect(self.show_vein_recognition)
        self.user_management_btn.clicked.connect(self.show_user_management)
        self.exit_btn.clicked.connect(self.parent.close)

    def show_vein_recognition(self):
        self.parent.stacked_widget.setCurrentIndex(1)

    def show_user_management(self):
        # 检查是否是管理员
        if not self.parent.is_admin:
            QMessageBox.warning(self, "权限不足", "只有管理员可以访问用户管理功能!")
            return

        # 如果用户管理界面还未创建，则创建它
        if not hasattr(self.parent, 'user_management_window'):
            from pages.user_management import UserManagementWindow
            self.parent.user_management_window = UserManagementWindow(self.parent)
            self.parent.stacked_widget.addWidget(self.parent.user_management_window)

        self.parent.stacked_widget.setCurrentWidget(self.parent.user_management_window)