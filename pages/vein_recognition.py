import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QComboBox, QInputDialog, QSpacerItem,
                             QSizePolicy, QSlider, QGroupBox)
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
        font = QFont("华文中宋", 12)

        # 图片显示区域
        self.image_label = QLabel()
        self.roi_label = QLabel()
        self.enhanced_label = QLabel()

        # 图片标题
        self.image_caption = QLabel("原图")
        self.roi_caption = QLabel("ROI")
        self.enhanced_caption = QLabel("图像增强")

        # 设置图片尺寸
        self.image_label.setFixedSize(350, 350)
        self.roi_label.setFixedSize(350, 350)
        self.enhanced_label.setFixedSize(200, 200)

        # 设置图片样式
        for label in [self.image_label, self.roi_label, self.enhanced_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid gray; background-color: rgba(255, 255, 255, 200)")

        # 设置标题样式
        for caption in [self.image_caption, self.roi_caption, self.enhanced_caption]:
            caption.setAlignment(Qt.AlignCenter)
            caption.setFont(font)
            caption.setStyleSheet("QLabel { color: black; }")

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

        # 主功能按钮
        self.upload_btn = QPushButton("上传图像")
        self.upload_btn.clicked.connect(self.upload_image)

        self.feature_btn = QPushButton("特征注册")
        self.feature_btn.clicked.connect(self.register_feature)

        self.match_btn = QPushButton("特征匹配")
        self.match_btn.clicked.connect(self.match_feature)

        self.back_btn = QPushButton("返回主菜单")
        self.back_btn.clicked.connect(self.go_back)

        # 模型选择下拉框
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Swin-transformer", "ResNet", "ViT", "MobileViT"])
        self.model_selector.setFont(font)
        self.model_selector.currentTextChanged.connect(self.change_model)

        # 距离度量选择下拉框
        self.metric_selector = QComboBox()
        self.metric_selector.addItems(["余弦距离", "欧式距离"])
        self.metric_selector.setFont(font)
        self.metric_selector.currentTextChanged.connect(self.change_metric)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.feature_btn)
        btn_layout.addWidget(self.match_btn)
        btn_layout.addWidget(self.model_selector)
        btn_layout.addWidget(self.metric_selector)
        btn_layout.addWidget(self.back_btn)

        # 图像布局
        image_layout = QHBoxLayout()

        # 原图列
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.image_label)
        vbox1.addWidget(self.image_caption)
        image_layout.addLayout(vbox1)

        # spacer between 图1 and 图2
        image_layout.addSpacerItem(QSpacerItem(50, 30, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # ROI列
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.roi_label)
        vbox2.addWidget(self.roi_caption)
        image_layout.addLayout(vbox2)

        # 图像增强列
        enhanced_container = QWidget()
        enhanced_vbox = QVBoxLayout(enhanced_container)

        # 添加参数控制
        param_label = QLabel("图像增强参数设置")
        param_label.setAlignment(Qt.AlignCenter)
        param_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        enhanced_vbox.addWidget(param_label)

        # 滑块控制
        slider_hbox = QHBoxLayout()

        # 对比度限制滑块
        self.clip_limit_slider = QSlider(Qt.Horizontal)
        self.clip_limit_slider.setRange(10, 80)  # 1.0-4.0
        self.clip_limit_slider.setValue(20)  # 默认2.0
        self.clip_limit_slider.valueChanged.connect(self.update_enhanced_image)

        # 网格大小滑块
        self.grid_size_slider = QSlider(Qt.Horizontal)
        self.grid_size_slider.setRange(1, 20)  # 1-20
        self.grid_size_slider.setValue(8)  # 默认8
        self.grid_size_slider.valueChanged.connect(self.update_enhanced_image)

        # 添加到滑块布局
        slider_hbox.addWidget(QLabel("对比度:"))
        slider_hbox.addWidget(self.clip_limit_slider)
        slider_hbox.addWidget(QLabel("网格:"))
        slider_hbox.addWidget(self.grid_size_slider)

        enhanced_vbox.addLayout(slider_hbox)

        # 添加增强图像和标题
        enhanced_vbox.addWidget(self.enhanced_label)
        enhanced_vbox.addWidget(self.enhanced_caption)

        image_layout.addWidget(enhanced_container)

        # 总体布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.result_text)

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