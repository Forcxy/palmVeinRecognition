from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QLineEdit, QGroupBox, QRadioButton, QCheckBox,
                             QMessageBox, QFormLayout, QProgressBar, QTextEdit, QApplication,
                             QSizePolicy, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QFont
from PyQt5.QtCore import Qt, pyqtSignal
import os
import cv2
import numpy as np
from core.roi_handler import extract_roi
from core.image_processor import enhance_image


class BatchROIWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        # 设置全局字体
        font = QFont("华文中宋", 10)
        QApplication.setFont(font)

        # 设置窗口背景和最小尺寸
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, Qt.lightGray)
        self.setPalette(palette)
        self.setMinimumWidth(700)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(15)

        # 统一样式表
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
            QSpinBox, QDoubleSpinBox {
                min-width: 80px;
            }
        """
        self.setStyleSheet(style)

        # 标题
        title_label = QLabel("批量提取ROI")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 30px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # 模式选择组
        mode_group = QGroupBox("")
        mode_layout = QFormLayout()

        # 单选按钮水平布局
        radio_layout = QHBoxLayout()
        self.single_files_radio = QRadioButton("批量处理文件")
        self.single_files_radio.setChecked(True)
        self.folder_radio = QRadioButton("递归处理文件夹")
        radio_layout.addWidget(self.single_files_radio)
        radio_layout.addWidget(self.folder_radio)
        radio_layout.addStretch()

        # 添加标签和单选按钮组
        mode_label = QLabel("选择模式")
        mode_label.setFixedWidth(80)
        mode_layout.addRow(mode_label, radio_layout)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # 输入设置组
        input_group = QGroupBox("")
        input_layout = QFormLayout()

        # 输入路径行
        input_label = QLabel("输入路径:")
        input_label.setFixedWidth(80)

        path_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("点击浏览按钮选择文件或文件夹")
        self.input_path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_files)

        path_layout.addWidget(self.input_path_edit)
        path_layout.addWidget(self.browse_btn)

        input_layout.addRow(input_label, path_layout)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # 输出设置组
        output_group = QGroupBox("")
        output_layout = QFormLayout()

        # 输出路径行
        output_label = QLabel("输出目录:")
        output_label.setFixedWidth(80)

        output_path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("点击浏览按钮选择输出目录")
        self.output_path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self.browse_output)

        output_path_layout.addWidget(self.output_path_edit)
        output_path_layout.addWidget(self.output_browse_btn)
        output_layout.addRow(output_label, output_path_layout)

        # 保持目录结构选项
        option_label = QLabel("目录选项:")
        option_label.setFixedWidth(80)
        self.keep_structure_check = QCheckBox("保持原始目录结构")
        self.keep_structure_check.setChecked(True)
        output_layout.addRow(option_label, self.keep_structure_check)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # 新增：图像处理参数设置组
        process_group = QGroupBox("")
        process_layout = QFormLayout()

        # 图像增强选项
        self.enhance_check = QCheckBox("启用图像增强")
        self.enhance_check.setChecked(True)
        process_layout.addRow(QLabel("图像增强:"), self.enhance_check)

        # 直方图均衡化参数
        histo_label = QLabel("直方图均衡化:")
        self.clip_limit_spin = QDoubleSpinBox()
        self.clip_limit_spin.setRange(1.0, 10.0)
        self.clip_limit_spin.setValue(2.0)
        self.clip_limit_spin.setSingleStep(0.5)
        self.clip_limit_spin.setPrefix("ClipLimit: ")

        self.tile_grid_spin = QSpinBox()
        self.tile_grid_spin.setRange(4, 32)
        self.tile_grid_spin.setValue(8)
        self.tile_grid_spin.setPrefix("TileGrid: ")

        histo_layout = QHBoxLayout()
        histo_layout.addWidget(self.clip_limit_spin)
        histo_layout.addWidget(self.tile_grid_spin)
        histo_layout.addStretch()

        process_layout.addRow(histo_label, histo_layout)

        # 高斯去噪选项
        self.denoise_check = QCheckBox("启用高斯去噪")
        self.denoise_check.setChecked(True)

        self.denoise_kernel_spin = QSpinBox()
        self.denoise_kernel_spin.setRange(3, 15)
        self.denoise_kernel_spin.setValue(5)
        self.denoise_kernel_spin.setSingleStep(2)
        self.denoise_kernel_spin.setPrefix("Kernel: ")
        #self.denoise_kernel_spin.setSuffix("x5")

        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(self.denoise_check)
        denoise_layout.addWidget(self.denoise_kernel_spin)
        denoise_layout.addStretch()

        process_layout.addRow(QLabel("高斯去噪:"), denoise_layout)

        process_group.setLayout(process_layout)
        main_layout.addWidget(process_group)

        # 按钮布局
        btn_layout = QHBoxLayout()

        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.process_files)
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

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 日志输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        self.setLayout(main_layout)

    def browse_files(self):
        if self.single_files_radio.isChecked():
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图像文件", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
            )
            if files:
                self.input_path_edit.setText(";".join(files))
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
            if folder:
                self.input_path_edit.setText(folder)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.output_path_edit.setText(folder)

    def process_files(self):
        input_path = self.input_path_edit.text()
        output_path = self.output_path_edit.text()

        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "请选择输入路径和输出目录!")
            return

        if self.single_files_radio.isChecked():
            files = input_path.split(";")
            self.process_file_list(files, output_path)
        else:
            self.process_folder(input_path, output_path)

    def process_file_list(self, files, output_dir):
        total = len(files)
        self.progress_bar.setMaximum(total)

        for i, file_path in enumerate(files):
            try:
                self.log_text.append(f"处理文件: {os.path.basename(file_path)}")

                # 读取图像
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log_text.append(f"  错误: 无法读取图像 {file_path}")
                    continue

                # 提取ROI
                _, roi_img = extract_roi(img)
                if roi_img is None:
                    self.log_text.append("  警告: 无法提取ROI")
                    continue

                # 图像增强处理
                processed_img = roi_img.copy()

                # 高斯去噪
                if self.denoise_check.isChecked():
                    kernel_size = self.denoise_kernel_spin.value()
                    processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)

                # 图像增强
                if self.enhance_check.isChecked():
                    clip_limit = self.clip_limit_spin.value()
                    tile_grid = self.tile_grid_spin.value()

                    # 创建CLAHE对象
                    clahe = cv2.createCLAHE(
                        clipLimit=clip_limit,
                        tileGridSize=(tile_grid, tile_grid)
                    )
                    processed_img = clahe.apply(processed_img)

                # 保存结果
                output_file = os.path.join(output_dir, os.path.basename(file_path))
                cv2.imwrite(output_file, processed_img)
                self.log_text.append(f"  结果已保存到: {output_file}")

            except Exception as e:
                self.log_text.append(f"  错误: {str(e)}")

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.log_text.append("批量处理完成!")

    def process_folder(self, input_dir, output_dir):
        # 获取所有图像文件
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        file_count = 0

        # 先统计文件总数用于进度条
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    file_count += 1

        if file_count == 0:
            self.log_text.append("警告: 未找到支持的图像文件!")
            return

        self.progress_bar.setMaximum(file_count)
        processed = 0

        for root, _, files in os.walk(input_dir):
            for file in files:
                if not file.lower().endswith(extensions):
                    continue

                try:
                    input_path = os.path.join(root, file)
                    self.log_text.append(f"处理文件: {input_path}")

                    # 读取图像（保持原始色彩空间）
                    img = cv2.imread(input_path)
                    if img is None:
                        self.log_text.append(f"  错误: 无法读取图像 {input_path}")
                        continue

                    # 提取ROI
                    _, roi_img = extract_roi(img)
                    if roi_img is None:
                        self.log_text.append("  警告: 无法提取ROI")
                        continue

                    # 获取处理参数
                    do_denoise = self.denoise_check.isChecked()
                    do_enhance = self.enhance_check.isChecked()

                    # 处理流程控制（与process_file_list相同）
                    processed_img = roi_img.copy()

                    if do_denoise and do_enhance:
                        kernel_size = self.denoise_kernel_spin.value()
                        processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)
                        if len(processed_img.shape) == 3:
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                        clip_limit = self.clip_limit_spin.value()
                        tile_grid = self.tile_grid_spin.value()
                        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
                        processed_img = clahe.apply(processed_img)
                    elif do_denoise and not do_enhance:
                        kernel_size = self.denoise_kernel_spin.value()
                        processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)
                    elif not do_denoise and do_enhance:
                        if len(processed_img.shape) == 3:
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                        clip_limit = self.clip_limit_spin.value()
                        tile_grid = self.tile_grid_spin.value()
                        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
                        processed_img = clahe.apply(processed_img)

                    # 保存结果
                    if self.keep_structure_check.isChecked():
                        rel_path = os.path.relpath(root, input_dir)
                        output_subdir = os.path.join(output_dir, rel_path)
                        os.makedirs(output_subdir, exist_ok=True)
                        output_path = os.path.join(output_subdir, file)
                    else:
                        output_path = os.path.join(output_dir, file)

                    cv2.imwrite(output_path, processed_img)
                    self.log_text.append(f"  结果已保存到: {output_path}")

                    processed += 1
                    self.progress_bar.setValue(processed)
                    QApplication.processEvents()

                except Exception as e:
                    self.log_text.append(f"  错误: {str(e)}")

        self.log_text.append(f"文件夹处理完成! 共处理 {processed}/{file_count} 个文件")

    def go_back(self):
        self.parent.stacked_widget.setCurrentIndex(0)