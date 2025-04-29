from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QComboBox, QSizePolicy,
                             QLineEdit, QFormLayout)
from PyQt5.QtGui import QFont, QBrush, QColor, QPalette
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core.database_handler import (load_features_by_model, get_user_features,
                                   delete_user_features, list_users)
import sqlite3
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 黑体，支持中文

class FeatureDatabaseWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.load_model_types()
        self.set_grey_background()

    def setup_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 标题
        title_label = QLabel("特征数据库管理")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
            }
        """)

        # 控制面板
        control_layout = QHBoxLayout()

        # 模型类型选择
        self.model_type_combo = QComboBox()
        self.model_type_combo.setPlaceholderText("选择模型类型")
        self.model_type_combo.setMinimumWidth(200)

        # 用户搜索框
        self.user_search_edit = QLineEdit()
        self.user_search_edit.setPlaceholderText("输入用户名搜索...")
        self.user_search_edit.setMinimumWidth(200)

        # 搜索按钮
        self.search_btn = QPushButton("搜索")
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        # 可视化按钮
        self.visualize_btn = QPushButton("可视化特征")
        self.visualize_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        # 删除按钮（仅管理员可见）
        self.delete_btn = QPushButton("删除特征")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.delete_btn.setVisible(self.parent.is_admin)

        # 返回按钮
        self.back_btn = QPushButton("返回主菜单")
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)

        # 第一行控制面板 - 模型选择和搜索
        top_control_layout = QHBoxLayout()
        top_control_layout.addWidget(QLabel("模型类型:"))
        top_control_layout.addWidget(self.model_type_combo)
        top_control_layout.addStretch(1)
        top_control_layout.addWidget(QLabel("用户搜索:"))
        top_control_layout.addWidget(self.user_search_edit)
        top_control_layout.addWidget(self.search_btn)

        # 第二行控制面板 - 操作按钮
        bottom_control_layout = QHBoxLayout()
        bottom_control_layout.addWidget(self.visualize_btn)
        bottom_control_layout.addWidget(self.delete_btn)
        bottom_control_layout.addStretch(1)
        bottom_control_layout.addWidget(self.back_btn)

        # 特征表格
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(4)
        self.feature_table.setHorizontalHeaderLabels(["用户名", "模型类型", "特征维度", "创建时间"])
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.feature_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # 可视化区域
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 添加到主布局
        main_layout.addWidget(title_label)
        main_layout.addLayout(top_control_layout)
        main_layout.addLayout(bottom_control_layout)
        main_layout.addWidget(self.feature_table)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

        # 连接信号
        self.model_type_combo.currentTextChanged.connect(self.load_features)
        self.search_btn.clicked.connect(self.search_user_features)
        self.user_search_edit.returnPressed.connect(self.search_user_features)
        self.visualize_btn.clicked.connect(self.visualize_features)
        self.delete_btn.clicked.connect(self.delete_selected_features)
        self.back_btn.clicked.connect(self.go_back)

    def set_grey_background(self):
        """设置灰色背景"""
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def load_model_types(self):
        """从数据库加载可用的模型类型"""
        self.model_type_combo.clear()
        conn = sqlite3.connect('feature_database.db')
        cursor = conn.cursor()

        # 获取所有不同的模型类型
        cursor.execute("SELECT DISTINCT model_type FROM features")
        model_types = [row[0] for row in cursor.fetchall()]

        # 如果没有模型类型，添加默认选项
        if not model_types:
            model_types = ["resnet", "swin", "viT", "mobileViT"]  # 默认模型类型

        self.model_type_combo.addItems(model_types)
        conn.close()

    def load_features(self, model_type):
        """加载指定模型类型的特征"""
        if not model_type:
            return

        conn = sqlite3.connect('feature_database.db')
        cursor = conn.cursor()

        # 获取指定模型类型的所有特征
        cursor.execute(
            "SELECT username, vector, created_at FROM features WHERE model_type = ?",
            (model_type,)
        )

        features = cursor.fetchall()
        self.feature_table.setRowCount(len(features))

        for row, (username, vector, created_at) in enumerate(features):
            # 用户名
            username_item = QTableWidgetItem(username)

            # 模型类型
            model_item = QTableWidgetItem(model_type)

            # 特征维度
            dim = len(np.frombuffer(vector, dtype=np.float32))
            dim_item = QTableWidgetItem(str(dim))

            # 创建时间
            time_item = QTableWidgetItem(created_at)

            self.feature_table.setItem(row, 0, username_item)
            self.feature_table.setItem(row, 1, model_item)
            self.feature_table.setItem(row, 2, dim_item)
            self.feature_table.setItem(row, 3, time_item)

        conn.close()

    def search_user_features(self):
        """搜索用户特征"""
        search_text = self.user_search_edit.text().strip()
        if not search_text:
            QMessageBox.warning(self, "提示", "请输入要搜索的用户名!")
            return

        model_type = self.model_type_combo.currentText()

        conn = sqlite3.connect('feature_database.db')
        cursor = conn.cursor()

        if model_type:
            # 搜索特定模型类型的用户特征
            cursor.execute(
                "SELECT username, model_type, vector, created_at FROM features WHERE username LIKE ? AND model_type = ?",
                (f"%{search_text}%", model_type)
            )
        else:
            # 搜索所有模型类型的用户特征
            cursor.execute(
                "SELECT username, model_type, vector, created_at FROM features WHERE username LIKE ?",
                (f"%{search_text}%",)
            )

        features = cursor.fetchall()
        self.feature_table.setRowCount(len(features))

        for row, (username, model_type, vector, created_at) in enumerate(features):
            # 用户名
            username_item = QTableWidgetItem(username)

            # 模型类型
            model_item = QTableWidgetItem(model_type)

            # 特征维度
            dim = len(np.frombuffer(vector, dtype=np.float32))
            dim_item = QTableWidgetItem(str(dim))

            # 创建时间
            time_item = QTableWidgetItem(created_at)

            self.feature_table.setItem(row, 0, username_item)
            self.feature_table.setItem(row, 1, model_item)
            self.feature_table.setItem(row, 2, dim_item)
            self.feature_table.setItem(row, 3, time_item)

        conn.close()

        if not features:
            QMessageBox.information(self, "提示", f"未找到用户 '{search_text}' 的特征数据")

    def visualize_features(self):
        """使用t-SNE可视化特征"""
        model_type = self.model_type_combo.currentText()
        if not model_type:
            QMessageBox.warning(self, "警告", "请先选择模型类型!")
            return

        features = load_features_by_model(model_type)
        if not features:
            QMessageBox.warning(self, "警告", "没有找到该类型的特征数据!")
            return

        # 准备数据
        usernames = list(features.keys())
        vectors = np.array(list(features.values()))

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(usernames) - 1))
        tsne_results = tsne.fit_transform(vectors)

        # 绘制图形
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 绘制点
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6)

        # 添加标签
        for i, username in enumerate(usernames):
            ax.text(tsne_results[i, 0], tsne_results[i, 1], username, fontsize=8)

        ax.set_title(f"{model_type} 特征向量 t-SNE 可视化")
        ax.set_xlabel("t-SNE 维度 1")
        ax.set_ylabel("t-SNE 维度 2")
        ax.grid(True)

        self.canvas.draw()

    def delete_selected_features(self):
        """删除选中的特征"""
        if not self.parent.is_admin:
            QMessageBox.warning(self, "权限不足", "只有管理员可以删除特征!")
            return

        selected = self.feature_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择要删除的特征!")
            return

        username = selected[0].text()
        model_type = selected[1].text()

        # 确认对话框
        reply = QMessageBox.question(
            self, '确认删除',
            f'确定要删除用户 "{username}" 的 {model_type} 特征吗?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            conn = sqlite3.connect('feature_database.db')
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM features WHERE username = ? AND model_type = ?",
                (username, model_type)
            )
            conn.commit()
            conn.close()

            if cursor.rowcount > 0:
                QMessageBox.information(self, "成功", f"用户 {username} 的 {model_type} 特征已删除")
                # 刷新当前视图
                if self.user_search_edit.text():
                    self.search_user_features()
                else:
                    self.load_features(model_type)
            else:
                QMessageBox.warning(self, "错误", "删除特征失败!")

    def go_back(self):
        """返回主菜单"""
        self.parent.stacked_widget.setCurrentIndex(0)