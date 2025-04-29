from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QLineEdit, QGroupBox, QRadioButton, QCheckBox,
                             QMessageBox, QFormLayout, QProgressBar, QTextEdit, QApplication,
                             QSizePolicy, QSpinBox, QDoubleSpinBox,QComboBox)
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QFont
from PyQt5.QtCore import Qt, pyqtSignal
import os
import cv2
import numpy as np
from core.roi_handler import extract_roi
from core.image_processor import enhance_image
from core.feature_extractor_Top import extract_features
from core.database_handler import save_feature


class BatchFeatureRegisterWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.image_files = []
        self.current_model = 'swin'
        self.image_type = 'original'
        self.processed_count = 0
        self.total_count = 0

    def setup_ui(self):
        # 设置全局字体
        font = QFont("华文中宋", 10)
        QApplication.setFont(font)

        # 设置窗口背景和最小尺寸（与BatchROIWindow一致）
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.setPalette(palette)
        self.setMinimumSize(800, 650)  # 比BatchROIWindow稍高以适应更多选项

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(15)

        # 统一样式表（与BatchROIWindow一致）
        style = """
            QLabel {
                text-align: left;
                padding-left: 5px;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 13px;
                padding: 0 5px;
            }
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
            QComboBox, QPushButton {
                min-height: 30px;
            }
        """
        self.setStyleSheet(style)

        # 标题（与BatchROIWindow一致）
        title_label = QLabel("批量特征注册")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 30px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # 模型和图像类型选择组
        settings_group = QGroupBox("")
        settings_layout = QFormLayout()

        # 模型选择
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Swin-transformer", "ResNet", "MobileViT", "ViT"])
        self.model_selector.currentTextChanged.connect(self.change_model)
        settings_layout.addRow("特征提取模型:", self.model_selector)

        # 图像类型选择
        self.image_type_selector = QComboBox()
        self.image_type_selector.addItems(["原始手掌图像", "已提取ROI图像"])
        self.image_type_selector.currentTextChanged.connect(self.change_image_type)
        settings_layout.addRow("图像类型:", self.image_type_selector)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # 输入设置组（与BatchROIWindow一致）
        input_group = QGroupBox("")
        input_layout = QFormLayout()

        # 输入路径行
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("点击浏览按钮选择文件夹")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_directory)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.input_path_edit)
        path_layout.addWidget(self.browse_btn)
        input_layout.addRow("图像目录:", path_layout)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # 按钮布局（与BatchROIWindow一致）
        btn_layout = QHBoxLayout()

        # 处理按钮
        self.process_btn = QPushButton("开始注册")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # 返回按钮
        self.back_btn = QPushButton("返回主菜单")
        self.back_btn.clicked.connect(self.go_back)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        btn_layout.addStretch()
        btn_layout.addWidget(self.process_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.back_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 进度条（与BatchROIWindow一致）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 日志输出（与BatchROIWindow一致）
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        self.setLayout(main_layout)

    def change_model(self, text):
        """切换特征提取模型"""
        if text == "Swin-transformer":
            self.current_model = 'swin'
        elif text == "ResNet":
            self.current_model = 'resnet'
        elif text == "ViT":
            self.current_model = 'viT'
        elif text == "MobileViT":
            self.current_model = 'mobileViT'
        self.log_text.append(f"[设置] 已选择模型: {text}")

    def change_image_type(self, text):
        self.image_type = 'original' if text == "原始手掌图像" else 'roi'
        self.log_text.append(f"[设置] 已选择图像类型: {text}")

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if dir_path:
            self.input_path_edit.setText(dir_path)
            self.scan_images(dir_path)

    def scan_images(self, dir_path):
        self.image_files = []
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(extensions):
                    self.image_files.append(os.path.join(root, file))

        self.total_count = len(self.image_files)
        self.progress_bar.setMaximum(self.total_count)

        if self.total_count > 0:
            self.process_btn.setEnabled(True)
            self.log_text.append(f"[扫描] 找到 {self.total_count} 张图像")
        else:
            self.process_btn.setEnabled(False)
            self.log_text.append("[警告] 未找到支持的图像文件!")

    def process_images(self):
        self.processed_count = 0
        self.log_text.append("[开始] 批量特征注册...")

        for image_path in self.image_files:
            try:
                # 读取图像
                img = cv2.imread(image_path)
                if img is None:
                    self.log_text.append(f"[错误] 无法读取图像: {os.path.basename(image_path)}")
                    continue

                # 根据图像类型处理
                if self.image_type == 'original':
                    # 处理原始手掌图像
                    _, roi_img = extract_roi(img)
                    if roi_img is None:
                        self.log_text.append(f"[警告] 无法提取ROI: {os.path.basename(image_path)}")
                        continue
                    processed_img = enhance_image(roi_img)
                else:
                    # 处理已提取的ROI图像
                    processed_img = enhance_image(img)

                # 提取特征
                feature = extract_features(
                    model_type=self.current_model,
                    image_input=processed_img,
                    roi_size=224
                ).numpy()

                # 使用文件名作为特征名
                feature_name = os.path.splitext(os.path.basename(image_path))[0]
                save_feature(feature_name, self.current_model, feature)

                self.log_text.append(f"[成功] 已注册: {feature_name}")
                self.processed_count += 1
                self.progress_bar.setValue(self.processed_count)

            except Exception as e:
                self.log_text.append(f"[错误] 处理失败 {os.path.basename(image_path)}: {str(e)}")

            QApplication.processEvents()

        self.log_text.append(f"[完成] 处理完成! 成功注册 {self.processed_count}/{self.total_count} 个特征")

    def go_back(self):
        self.parent.stacked_widget.setCurrentIndex(0)