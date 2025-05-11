import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QComboBox, QInputDialog, QSpacerItem,
                             QSizePolicy, QSlider, QGroupBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from core.roi_handler import extract_roi
from core.image_processor import enhance_image
from core.database_handler import save_feature, load_features_by_model
from core.feature_extractor_Top import extract_features


def cosine_similarity(a, b):
    """余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    """欧式距离"""
    return np.linalg.norm(a - b)


class VeinRecognitionWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.original_image = None
        self.enhanced_image = None
        self.roi_image = None
        self.detect_image = None
        self.current_model = 'swin'
        self.current_metric = 'cosine'
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = 8

    def setup_ui(self):
        # 设置主窗口背景为带白色线条纹理的蓝色背景
        # self.setStyleSheet("""
        #     QWidget {
        #         background-color: qlineargradient(
        #             spread:pad, x1:0, y1:0, x2:1, y2:1,
        #             stop:0 rgba(30, 70, 120, 255),
        #             stop:1 rgba(50, 100, 160, 255)
        #         );
        #         background-image: url(:/texture.png);
        #     }
        # """)

        # 主字体设置 - 使用华文中宋
        font = QFont("华文中宋", 12)
        title_font = QFont("华文中宋", 12, QFont.Bold)

        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 创建顶部按钮栏
        self.create_top_button_bar(main_layout, font)

        # 创建图像显示区域
        self.create_image_display_area(main_layout, font, title_font)

        # 创建结果文本框
        self.create_result_textbox(main_layout, font)

    def create_top_button_bar(self, parent_layout, font):
        """创建顶部功能按钮栏"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # 上传按钮
        self.upload_btn = QPushButton("上传图像")
        self.upload_btn.setFont(font)
        self.upload_btn.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_btn)

        # 特征注册按钮
        self.feature_btn = QPushButton("特征注册")
        self.feature_btn.setFont(font)
        self.feature_btn.clicked.connect(self.register_feature)
        button_layout.addWidget(self.feature_btn)

        # 特征匹配按钮
        self.match_btn = QPushButton("特征匹配")
        self.match_btn.setFont(font)
        self.match_btn.clicked.connect(self.match_feature)
        button_layout.addWidget(self.match_btn)

        # 模型选择下拉框
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Swin-transformer", "ResNet", "ViT", "MobileViT"])
        self.model_selector.setFont(font)
        self.model_selector.setStyleSheet("""
            QComboBox {
                background-color: rgba(255,255,255,150);
                border: 1px solid gray;
                border-radius: 5px;
                padding: 3px;
                min-width: 120px;
            }
        """)
        self.model_selector.currentTextChanged.connect(self.change_model)
        button_layout.addWidget(self.model_selector)

        # 距离度量选择下拉框
        self.metric_selector = QComboBox()
        self.metric_selector.addItems(["余弦距离", "欧式距离"])
        self.metric_selector.setFont(font)
        self.metric_selector.setStyleSheet(self.model_selector.styleSheet())
        self.metric_selector.currentTextChanged.connect(self.change_metric)
        button_layout.addWidget(self.metric_selector)

        # 返回按钮
        self.back_btn = QPushButton("返回主菜单")
        self.back_btn.setFont(font)
        self.back_btn.clicked.connect(self.go_back)
        button_layout.addWidget(self.back_btn)

        # 设置按钮样式
        for btn in [self.upload_btn, self.feature_btn, self.match_btn, self.back_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(70,130,180,200);
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 15px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: rgba(70,130,180,150);
                }
            """)

        parent_layout.addLayout(button_layout)

    def create_image_display_area(self, parent_layout, font, title_font):
        """创建图像显示区域"""
        image_layout = QHBoxLayout()
        image_layout.setSpacing(20)

        # 原图区域
        self.create_image_panel(image_layout, "原图", 350, font)

        # ROI区域
        self.create_image_panel(image_layout, "ROI", 350, font)

        # 图像增强区域
        self.create_enhancement_panel(image_layout, title_font)

        parent_layout.addLayout(image_layout)

    def create_image_panel(self, parent_layout, title, size, font):
        """创建单个图像显示面板（外层无边框+透明，内层保留白色背景）"""
        container = QFrame()
        # container.setStyleSheet("""
        #     QFrame {
        #         background-color: transparent;  # 外层透明
        #         border: none;                   # 外层无边框
        #     }
        # """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除外层内边距
        layout.setSpacing(5)  # 标题和图像之间的间距

        # 图像标签（保留白色背景，但无额外边框）
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setFixedSize(size, size)
        image_label.setStyleSheet("""
            QLabel {
                background-color: rgba(240,240,240,180);
                border-radius: 8px;
                border: 1px solid gray;
            }
        """)
        layout.addWidget(image_label)

        # 图像标题（仅文字，无背景）
        caption = QLabel(title)
        caption.setAlignment(Qt.AlignCenter)
        caption.setFont(font)
        caption.setStyleSheet("QLabel { color: black; background: transparent; }")
        layout.addWidget(caption)

        # 保存引用
        if title == "原图":
            self.image_label = image_label
            self.image_caption = caption
        elif title == "ROI":
            self.roi_label = image_label
            self.roi_caption = caption

        parent_layout.addWidget(container)

    def create_enhancement_panel(self, parent_layout, title_font):
        """创建图像增强面板（外层无边框+透明，内层保留白色背景）"""
        container = QFrame()
        # container.setStyleSheet("""
        #     QFrame {
        #         background-color: transparent;  # 外层透明
        #         border: none;                   # 外层无边框
        #     }
        # """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除外层内边距
        layout.setSpacing(10)  # 内部组件间距

        # 参数设置标题
        param_label = QLabel("图像增强参数设置")
        param_label.setAlignment(Qt.AlignCenter)
        param_label.setFont(title_font)
        param_label.setStyleSheet("QLabel { color: white; margin-bottom: 10px; font-weight: normal;}")  # 改为黑色文字
        layout.addWidget(param_label)

        # 对比度控制
        contrast_layout = QHBoxLayout()
        contrast_layout.setSpacing(10)

        contrast_label = QLabel("对比度阈值:")
        contrast_label.setFont(title_font)
        contrast_label.setStyleSheet("QLabel { color: white; min-width: 60px; font-weight: normal;}")  # 改为黑色文字
        contrast_layout.addWidget(contrast_label)

        self.clip_limit_slider = QSlider(Qt.Horizontal)
        self.clip_limit_slider.setRange(10, 40)
        self.clip_limit_slider.setValue(20)
        self.clip_limit_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(80, 80, 80, 150);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: rgb(70, 130, 180);
                border: 2px solid rgb(80, 140, 200);
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 1px;  /* 轻微圆角 */
            }
            QSlider::sub-page:horizontal {
                background: rgb(70, 130, 180);
                border-radius: 3px;
            }
        """)
        self.clip_limit_slider.valueChanged.connect(self.update_enhanced_image)
        contrast_layout.addWidget(self.clip_limit_slider, 1)

        self.clip_limit_value = QLabel("2.0")
        self.clip_limit_value.setFont(title_font)
        self.clip_limit_value.setStyleSheet("QLabel { color: white; min-width: 30px; font-weight: normal;}")  # 改为黑色文字
        self.clip_limit_value.setAlignment(Qt.AlignCenter)
        contrast_layout.addWidget(self.clip_limit_value)

        layout.addLayout(contrast_layout)

        # 网格控制
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(10)

        grid_label = QLabel("网格:")
        grid_label.setFont(title_font)
        grid_label.setStyleSheet("QLabel { color: white; min-width: 60px; font-weight: normal;}")  # 改为黑色文字
        grid_layout.addWidget(grid_label)

        self.grid_size_slider = QSlider(Qt.Horizontal)
        self.grid_size_slider.setRange(1, 20)
        self.grid_size_slider.setValue(8)
        self.grid_size_slider.setStyleSheet(self.clip_limit_slider.styleSheet())
        self.grid_size_slider.valueChanged.connect(self.update_enhanced_image)
        grid_layout.addWidget(self.grid_size_slider, 1)

        self.grid_size_value = QLabel("8")
        self.grid_size_value.setFont(title_font)
        self.grid_size_value.setStyleSheet("QLabel { color: white; min-width: 30px; font-weight: normal;}")  # 改为黑色文字
        self.grid_size_value.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.grid_size_value)

        layout.addLayout(grid_layout)

        # 增强图像显示（保留白色背景）
        self.enhanced_label = QLabel()
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setFixedSize(200, 200)
        self.enhanced_label.setStyleSheet("""
            QLabel {
                background-color: rgba(240,240,240,180);
                border-radius: 8px;
                border: 1px solid gray;
                margin-top: 15px;
                margin-bottom: 5px;
            }
        """)
        layout.addWidget(self.enhanced_label, 0, Qt.AlignHCenter)

        # 增强图像标题
        self.enhanced_caption = QLabel("图像增强")
        self.enhanced_caption.setAlignment(Qt.AlignCenter)
        self.enhanced_caption.setFont(title_font)
        self.enhanced_caption.setStyleSheet("QLabel { color: black; font-weight: normal;}")  # 改为黑色文字
        layout.addWidget(self.enhanced_caption)

        parent_layout.addWidget(container)

    def create_result_textbox(self, parent_layout, font):
        """创建结果文本框"""
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(font)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid #999;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        parent_layout.addWidget(self.result_text)

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

        self.result_text.append(f"[INFO] 已切换模型: {text}")

    def change_metric(self, text):
        """切换距离度量方法"""
        if text == "余弦距离":
            self.current_metric = 'cosine'
        else:
            self.current_metric = 'euclidean'
        self.result_text.append(f"[INFO] 已切换距离度量: {text}")

    def go_back(self):
        self.parent.stacked_widget.setCurrentIndex(0)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.original_image = cv2.imread(file_path)
        self.display_image(self.image_label, self.original_image)

        self.detect_image, self.roi_image = extract_roi(self.original_image)
        if self.roi_image is not None:
            self.display_image(self.roi_label, self.detect_image)
            self.update_enhanced_image()
            self.result_text.append("[INFO] 图像加载并处理成功")
        else:
            self.result_text.append("[WARN] ROI提取失败，请上传清晰的手掌图像")

    def display_image(self, label, img):
        """支持显示彩色和灰度图像"""
        if len(img.shape) == 3:  # 彩色图像 (BGR)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:  # 灰度图像
            h, w = img.shape
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def enhance_image(self, image, clip_limit=2.0, grid_size=8):
        """增强图像(带CLAHE参数)"""
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size)
        )

        if len(image.shape) == 3:  # 彩色图像
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        else:  # 灰度图像
            return clahe.apply(image)

    def update_enhanced_image(self):
        """当CLAHE参数变化时更新增强图像"""
        if self.roi_image is not None:
            clip_limit = self.clip_limit_slider.value() / 10.0
            grid_size = self.grid_size_slider.value()

            # 更新数值显示
            self.clip_limit_value.setText(f"{clip_limit:.1f}")
            self.grid_size_value.setText(f"{grid_size}")

            self.enhanced_image = self.enhance_image(
                self.roi_image,
                clip_limit=clip_limit,
                grid_size=grid_size
            )

            self.display_image(self.enhanced_label, self.enhanced_image)

    def register_feature(self):
        if self.roi_image is not None:
            vec = extract_features(
                model_type=self.current_model,
                image_input=self.enhanced_image,
                roi_size=224
            ).numpy()

            name, ok = QInputDialog.getText(self, "用户注册", "请输入用户名：")
            if ok and name:
                user = os.path.splitext(os.path.basename(name))[0]
                save_feature(user, self.current_model, vec)
                self.result_text.append(f"[INFO] {self.current_model}特征向量已注册为：{user}")

    def match_feature(self):
        if self.roi_image is not None:
            current_vec = extract_features(
                model_type=self.current_model,
                image_input=self.enhanced_image,
                roi_size=224
            ).numpy()

            db = load_features_by_model(self.current_model)
            if not db:
                self.result_text.append(f"[WARN] 没有找到{self.current_model}模型的注册特征")
                return

            scores = []
            for user, db_vec in db.items():
                if self.current_metric == 'cosine':
                    score = cosine_similarity(current_vec, db_vec)
                else:
                    score = -euclidean_distance(current_vec, db_vec)
                scores.append((user, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            metric_name = "相似度" if self.current_metric == 'cosine' else "距离(负值)"
            self.result_text.append(f"\n[MATCH] {self.current_model}模型匹配结果（{metric_name}）排名：")
            for i, (user, score) in enumerate(scores[:3], 1):
                self.result_text.append(f"TOP {i}: {user} ({metric_name}: {score:.4f})")

            best_user, best_score = scores[0]
            if self.current_metric == 'cosine':
                if best_score > 0.8:
                    self.result_text.append(f"\n[RESULT] 匹配成功: {best_user}")
                else:
                    self.result_text.append("\n[RESULT] 无可靠匹配")
            else:
                if -best_score < 1.0:
                    self.result_text.append(f"\n[RESULT] 匹配成功: {best_user}")
                else:
                    self.result_text.append("\n[RESULT] 无可靠匹配")