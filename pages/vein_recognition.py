import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
                           QHBoxLayout, QTextEdit, QComboBox, QInputDialog,QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from core.roi_handler import extract_roi
from core.image_processor import enhance_image
from core.database_handler import save_feature, load_feature
from core.feature_extractor import extract_swin_features

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def dummy_model_feature(img):
    model_weight_path = r"C:\Users\CXY\Desktop\graduationDesign\src\palmVeinRecognition\application\weights\model_swint56.pth"
    return extract_swin_features(
        image_input=img,  # 直接传递numpy数组
        weight_path=model_weight_path,
        roi_size=224
    )


class VeinRecognitionWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.original_image = None
        self.enhanced_image = None
        self.roi_image = None
        self.detect_image = None

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

        self.upload_btn = QPushButton("上传图像")
        self.upload_btn.clicked.connect(self.upload_image)

        self.feature_btn = QPushButton("特征注册")
        self.feature_btn.clicked.connect(self.register_feature)

        self.match_btn = QPushButton("特征匹配")
        self.match_btn.clicked.connect(self.match_feature)

        self.back_btn = QPushButton("返回主菜单")
        self.back_btn.clicked.connect(self.go_back)

        for btn in [self.upload_btn, self.feature_btn, self.match_btn, self.back_btn]:
            btn.setFont(font)
            btn.setStyleSheet(
                "QPushButton { background-color: #ADD8E6; color: black; padding: 10px; border-radius: 10px; } QPushButton:hover { background-color: #87CEEB; }")

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Swin-transformer"])
        self.model_selector.setFont(font)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.feature_btn)
        btn_layout.addWidget(self.match_btn)
        btn_layout.addWidget(self.model_selector)
        btn_layout.addWidget(self.back_btn)

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

        # 图像增强列（垂直居中处理）
        enhanced_vbox = QVBoxLayout()
        enhanced_vbox.addStretch(1)
        enhanced_vbox.addWidget(self.enhanced_label, alignment=Qt.AlignHCenter)
        enhanced_vbox.addStretch(1)
        enhanced_vbox.addWidget(self.enhanced_caption)
        image_layout.addLayout(enhanced_vbox)

        # 总体布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.result_text)

        self.setLayout(main_layout)




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
            self.enhanced_image = enhance_image(self.roi_image)
            self.display_image(self.enhanced_label, self.enhanced_image)
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
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def register_feature(self):
        if self.roi_image is not None:
            vec = dummy_model_feature(self.enhanced_image)
            name, ok = QInputDialog.getText(self, "用户注册", "请输入用户名：")
            if ok and name:
                user = os.path.splitext(os.path.basename(name))[0]
                save_feature(user, vec)
                self.result_text.append(f"[INFO] 特征向量已注册为：{user}")

    def match_feature(self):
        if self.roi_image is not None:
            # 1. 提取当前图像特征
            current_vec = dummy_model_feature(self.enhanced_image)

            # 2. 加载数据库
            db = load_feature()
            if not db:
                self.result_text.append("[WARN] 数据库为空，无法匹配")
                return

            # 打印所有姓名和特征向量
            self.result_text.append("[INFO] 数据库中的注册用户：")
            for name, vector in db.items():
                self.result_text.append(f"  - 用户名: {name}")
                self.result_text.append(f"    特征向量: {vector.tolist()[:5]}... (共 {len(vector)} 维)")

            # 3. 计算所有相似度并排序
            similarity_scores = []
            for user, db_vec in db.items():
                sim = cosine_similarity(current_vec, np.array(db_vec))
                similarity_scores.append((user, sim))

            # 按相似度降序排序
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # 4. 显示前三结果
            self.result_text.append("\n[MATCH] 相似度排名：")
            for i, (user, score) in enumerate(similarity_scores[:3], 1):
                self.result_text.append(f"TOP {i}: {user} (相似度: {score:.4f})")

            # 5. 添加匹配建议
            best_user, best_score = similarity_scores[0]
            if best_score > 0.8:  # 假设阈值设为0.8
                self.result_text.append(f"\n[RESULT] 匹配成功: {best_user}")
            else:
                self.result_text.append("\n[RESULT] 无可靠匹配")