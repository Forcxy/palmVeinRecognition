from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QTableWidget, QTableWidgetItem,
                            QHeaderView, QMessageBox, QInputDialog)
from PyQt5.QtGui import QFont, QBrush, QColor
from PyQt5.QtCore import Qt
from core.database_handler import list_users, delete_user, is_admin

class UserManagementWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.load_users()
        
    def setup_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("用户管理")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        
        # 用户表格
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(4)
        self.user_table.setHorizontalHeaderLabels(["用户名", "角色", "创建时间", "最后登录"])
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.user_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("刷新")
        self.delete_btn = QPushButton("删除用户")
        self.back_btn = QPushButton("返回主菜单")
        
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
            QPushButton#delete_btn {
                background-color: #e74c3c;
            }
            QPushButton#delete_btn:hover {
                background-color: #c0392b;
            }
        """
        self.refresh_btn.setStyleSheet(button_style)
        self.delete_btn.setObjectName("delete_btn")
        self.delete_btn.setStyleSheet(button_style)
        self.back_btn.setStyleSheet(button_style)
        
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(self.back_btn)
        
        # 添加到主布局
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.user_table)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # 连接信号
        self.refresh_btn.clicked.connect(self.load_users)
        self.delete_btn.clicked.connect(self.delete_selected_user)
        self.back_btn.clicked.connect(self.go_back)
        
    def load_users(self):
        users = list_users()
        self.user_table.setRowCount(len(users))
        
        for row, user in enumerate(users):
            # 用户名
            username_item = QTableWidgetItem(user["username"])
            if user["is_admin"]:
                username_item.setForeground(QBrush(QColor(0, 128, 0)))  # 管理员绿色
            
            # 角色
            role_item = QTableWidgetItem("管理员" if user["is_admin"] else "普通用户")
            
            # 创建时间
            created_item = QTableWidgetItem(user["created_at"])
            
            # 最后登录
            last_login = user["last_login"] or "从未登录"
            last_login_item = QTableWidgetItem(last_login)
            
            # 添加到表格
            self.user_table.setItem(row, 0, username_item)
            self.user_table.setItem(row, 1, role_item)
            self.user_table.setItem(row, 2, created_item)
            self.user_table.setItem(row, 3, last_login_item)
            
    def delete_selected_user(self):
        selected = self.user_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择要删除的用户!")
            return
            
        username = selected[0].text()
        
        if is_admin(username):
            QMessageBox.warning(self, "错误", "不能删除管理员账户!")
            return
            
        # 确认对话框
        reply = QMessageBox.question(
            self, '确认删除',
            f'确定要删除用户 "{username}" 吗? 此操作不可恢复!',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            if delete_user(username):
                QMessageBox.information(self, "成功", f"用户 {username} 已删除")
                self.load_users()
            else:
                QMessageBox.warning(self, "错误", "删除用户失败!")
                
    def go_back(self):
        self.parent.stacked_widget.setCurrentIndex(0)