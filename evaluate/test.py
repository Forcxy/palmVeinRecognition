import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple, Dict
from core.feature_extractor import extract_swin_features
import matplotlib.pyplot as plt
import cv2
from core.image_processor import enhance_image
import warnings

warnings.filterwarnings("ignore")

# 全局设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


class CASIADatasetProcessor:
    def __init__(self, dataset_path: str, weight_path: str, device: str = None):
        """初始化数据集处理器"""
        self.dataset_path = dataset_path
        self.weight_path = weight_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = sorted([d for d in os.listdir(dataset_path)
                               if os.path.isdir(os.path.join(dataset_path, d))])
        self.features_cache = {}
        self._validate_dataset()

    def _validate_dataset(self):
        """验证数据集结构完整性"""
        print("Validating dataset structure...")
        for cls in self.classes:
            cls_path = os.path.join(self.dataset_path, cls)
            if len(os.listdir(cls_path)) < 2:
                raise ValueError(f"Class {cls} has insufficient images (minimum 2 required)")
        print(f"Dataset validated. Total classes: {len(self.classes)}")

    def extract_all_features(self, force_reload: bool = False):
        """批量提取特征向量"""
        if not force_reload and self.features_cache:
            print("Using cached features")
            return

        print("Extracting features...")
        self.features_cache = {}

        for cls in tqdm(self.classes, desc="Processing"):
            cls_path = os.path.join(self.dataset_path, cls)
            features = []

            for img_name in sorted(os.listdir(cls_path)):
                try:
                    img = cv2.imread(os.path.join(cls_path, img_name))
                    if img is None:
                        continue

                    # enhanced = enhance_image(img)
                    feat = extract_swin_features(
                        image_input=img,
                        weight_path=self.weight_path,
                        device=self.device
                    )
                    features.append(feat)
                except Exception as e:
                    print(f"Error processing {img_name}: {str(e)}")
                    continue

            if features:
                self.features_cache[cls] = torch.stack(features)
        print("Feature extraction completed.")

    def build_pairs(self, num_pairs: int = 1000) -> Tuple[List, List]:
        """构建正负样本对"""
        self.extract_all_features()
        n_classes = len(self.classes)
        m_images = len(next(iter(self.features_cache.values())))

        print(f"\nBuilding {num_pairs} positive/negative pairs...")

        # 正样本对（同类不同图像）
        pos_pairs = [
            (random.randint(0, n_classes - 1), *random.sample(range(m_images), 2))
            for _ in range(num_pairs)
        ]

        # 负样本对（不同类随机图像）
        neg_pairs = [
            (*random.sample(range(n_classes), 2),
             random.randint(0, m_images - 1),
             random.randint(0, m_images - 1))
            for _ in range(num_pairs)
        ]

        return pos_pairs, neg_pairs

    def compute_similarities(self, pos_pairs: List, neg_pairs: List) -> Tuple[np.ndarray, np.ndarray]:
        """计算样本对相似度"""
        print("\nComputing similarities...")

        def get_sim(pair, is_pos=True):
            cls_idx, img1, img2 = pair[:3] if is_pos else (pair[0], pair[2], pair[3])
            feat1 = self.features_cache[self.classes[cls_idx]][img1]
            feat2 = self.features_cache[self.classes[img2 if is_pos else pair[1]]][img2]
            return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

        pos_sims = [get_sim(p) for p in tqdm(pos_pairs, desc="Positive pairs")]
        neg_sims = [get_sim(p, False) for p in tqdm(neg_pairs, desc="Negative pairs")]

        return np.array(pos_sims), np.array(neg_sims)

    def _compute_metrics(self, pos_sims: np.ndarray, neg_sims: np.ndarray) -> Dict:
        """核心指标计算"""
        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_score = np.concatenate([pos_sims, neg_sims])

        # ROC曲线计算
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # EER计算
        frr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(frr - fpr))
        eer = frr[eer_idx]
        best_threshold = thresholds[eer_idx]

        # CRR计算
        pos_correct = np.sum(pos_sims >= best_threshold)
        neg_correct = np.sum(neg_sims < best_threshold)
        crr = (pos_correct + neg_correct) / (len(pos_sims) + len(neg_sims))

        return {
            'fpr': fpr,
            'frr': frr,
            'auc': roc_auc,
            'eer': eer,
            'threshold': best_threshold,
            'crr': crr,
            'pos_acc': pos_correct / len(pos_sims),
            'neg_acc': neg_correct / len(neg_sims)
        }

    def _plot_metrics(self, fpr: np.ndarray, frr: np.ndarray, auc: float, eer: float):
        """绘制ERR曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, frr, color='darkorange', lw=2,
                 label=f'ERR Curve (AUC = {auc:.3f})')
        plt.scatter(eer, eer, color='red', s=100, label=f'EER = {eer:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('Performance Evaluation')
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate(self, num_pairs: int = 1000) -> Dict:
        """完整评估流程"""
        # 1. 数据准备
        pos_pairs, neg_pairs = self.build_pairs(num_pairs)
        pos_sims, neg_sims = self.compute_similarities(pos_pairs, neg_pairs)

        # 2. 指标计算
        metrics = self._compute_metrics(pos_sims, neg_sims)

        # 3. 结果展示
        print("\n" + "=" * 40)
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"EER: {metrics['eer']:.4f} (Threshold={metrics['threshold']:.4f})")
        print(f"CRR: {metrics['crr']:.4f}")
        print(f"Positive Accuracy: {metrics['pos_acc']:.4f}")
        print(f"Negative Accuracy: {metrics['neg_acc']:.4f}")
        print("=" * 40)

        # 4. 可视化
        self._plot_metrics(metrics['fpr'], metrics['frr'], metrics['auc'], metrics['eer'])

        return metrics


if __name__ == "__main__":
    # 配置参数
    DATASET_PATH = "../dataset/CASIA"
    WEIGHT_PATH = "../weights/model_swint56.pth"

    # 执行评估
    processor = CASIADatasetProcessor(DATASET_PATH, WEIGHT_PATH)
    results = processor.evaluate(num_pairs=1000)