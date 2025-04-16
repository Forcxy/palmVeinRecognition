import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from typing import List, Tuple, Dict
from core.feature_extractor import extract_swin_features
import matplotlib.pyplot as plt


# 设置字体
plt.rcParams['font.family'] = ['Times New Roman'] # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class CASIADatasetProcessor:
    def __init__(self, dataset_path: str, weight_path: str, device: str = None):
        """
        初始化CASIA掌静脉数据集处理器

        参数:
            dataset_path: 数据集路径 (包含子文件夹的结构)
            weight_path: 模型权重路径
            device: 计算设备 (cuda/cpu)
        """
        self.dataset_path = dataset_path
        self.weight_path = weight_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.classes = sorted(os.listdir(dataset_path)) 去掉—_2子类
        self.classes = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d)) and not d.endswith('_2')
        ])
        self.features_cache = {}  # 缓存特征向量
        self._validate_dataset()

    def _validate_dataset(self):
        """验证数据集结构是否符合预期"""
        print("验证数据集结构...")
        for cls in self.classes:
            cls_path = os.path.join(self.dataset_path, cls)
            if not os.path.isdir(cls_path):
                raise ValueError(f"无效的类别路径: {cls_path}")

            images = os.listdir(cls_path)
            if len(images) < 2:
                raise ValueError(f"类别 {cls} 中的图像数量不足 (至少需要2张)")
        print(f"数据集验证通过，共 {len(self.classes)} 个类别")

    def extract_all_features(self, force_reload: bool = False):
        """
        提取所有图像的特征向量并缓存

        参数:
            force_reload: 是否强制重新提取特征 (忽略缓存)
        """
        if not force_reload and self.features_cache:
            print("使用缓存的特征向量")
            return

        print("开始提取所有图像的特征向量...")
        self.features_cache = {}

        for cls in tqdm(self.classes, desc="提取特征"):
            cls_path = os.path.join(self.dataset_path, cls)
            images = sorted(os.listdir(cls_path))
            cls_features = []

            for img_name in images:
                img_path = os.path.join(cls_path, img_name)
                feature = extract_swin_features(
                    img_path,
                    self.weight_path,
                    device=self.device
                )
                cls_features.append(feature)

            self.features_cache[cls] = torch.stack(cls_features)

        print("特征提取完成")

    def build_pairs(self, custom_num_pairs: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        构建正负样本对

        参数:
            custom_num_pairs: 自定义正负样本对数量 (None则自动计算)

        返回:
            positive_pairs: 正样本对列表 [(class_idx, img_idx1, img_idx2), ...]
            negative_pairs: 负样本对列表 [(class_idx1, img_idx1, class_idx2, img_idx2), ...]
        """
        self.extract_all_features()

        # 计算默认的正负样本对数量
        n_classes = len(self.classes)
        m_images = len(next(iter(self.features_cache.values())))  # 每个类别的图像数量

        if custom_num_pairs is None:
            num_pairs = n_classes * m_images * (m_images - 1) // 2
        else:
            num_pairs = custom_num_pairs

        print(f"\n构建样本对 (目标数量: {num_pairs})")
        print(f"类别数: {n_classes}, 每类图像数: {m_images}")

        # 构建正样本对 (同一类别内的所有可能组合)
        positive_pairs = []
        for cls_idx, cls in enumerate(self.classes):
            img_indices = list(range(m_images))
            for img1, img2 in combinations(img_indices, 2):
                positive_pairs.append((cls_idx, img1, img2))

        # 随机选择等量的负样本对
        negative_pairs = []
        for _ in range(num_pairs):
            # 随机选择两个不同类别
            cls1_idx, cls2_idx = random.sample(range(n_classes), 2)
            cls1, cls2 = self.classes[cls1_idx], self.classes[cls2_idx]

            # 从每个类别中随机选择一张图像
            img1_idx = random.randint(0, m_images - 1)
            img2_idx = random.randint(0, m_images - 1)

            negative_pairs.append((cls1_idx, img1_idx, cls2_idx, img2_idx))

        # 如果自定义数量小于自动计算的正样本对数量，则随机采样
        if custom_num_pairs is not None and custom_num_pairs < len(positive_pairs):
            positive_pairs = random.sample(positive_pairs, custom_num_pairs)
            negative_pairs = negative_pairs[:custom_num_pairs]

        print(f"正样本对数量: {len(positive_pairs)}")
        print(f"负样本对数量: {len(negative_pairs)}")

        return positive_pairs, negative_pairs

    def compute_similarities(self, positive_pairs: List, negative_pairs: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算正负样本对的余弦相似度

        参数:
            positive_pairs: 正样本对列表
            negative_pairs: 负样本对列表

        返回:
            pos_sims: 正样本对相似度数组
            neg_sims: 负样本对相似度数组
        """
        print("\n计算余弦相似度...")

        # 计算正样本对相似度
        pos_sims = []
        for cls_idx, img1_idx, img2_idx in tqdm(positive_pairs, desc="正样本对"):
            cls = self.classes[cls_idx]
            feat1 = self.features_cache[cls][img1_idx]
            feat2 = self.features_cache[cls][img2_idx]
            sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
            pos_sims.append(sim)

        # 计算负样本对相似度
        neg_sims = []
        for cls1_idx, img1_idx, cls2_idx, img2_idx in tqdm(negative_pairs, desc="负样本对"):
            cls1 = self.classes[cls1_idx]
            cls2 = self.classes[cls2_idx]
            feat1 = self.features_cache[cls1][img1_idx]
            feat2 = self.features_cache[cls2][img2_idx]
            sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
            neg_sims.append(sim)

        # 缓存结果用于后续绘图
        self.last_pos_sims = np.array(pos_sims)
        self.last_neg_sims = np.array(neg_sims)

        return self.last_pos_sims, self.last_neg_sims

    def evaluate_performance(self, pos_sims: np.ndarray, neg_sims: np.ndarray, plot: bool = True):
        """
        评估性能并绘制ROC曲线和ERR曲线
        """
        print("\n评估性能...")
        print(f"正样本对数量: {len(pos_sims)}")
        print(f"负样本对数量: {len(neg_sims)}")

        # 准备标签和预测分数
        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_score = np.concatenate([pos_sims, neg_sims])

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 计算FRR (False Rejection Rate) = 1 - TPR
        frr = 1 - tpr

        # 计算EER (Equal Error Rate)
        eer_idx = np.nanargmin(np.absolute(frr - fpr))
        eer = frr[eer_idx]
        eer_threshold = thresholds[eer_idx]

        print(f"ROC曲线下面积 (AUC): {roc_auc:.4f}")
        print(f"等错误率 (EER): {eer:.4f} (阈值={eer_threshold:.4f})")

        if plot:
            # 只需要传递 fpr, tpr, roc_auc, eer
            self._plot_curves(fpr, tpr, roc_auc, eer)

    def _plot_curves(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, eer: float):
        """绘制ROC曲线和ERR曲线
        参数:
            fpr: 假阳性率 (False Positive Rate)
            tpr: 真阳性率 (True Positive Rate)
            roc_auc: ROC曲线下面积
            eer: 等错误率
        """
        plt.figure(figsize=(12, 5))

        # 计算FRR (False Rejection Rate = 1 - TPR)
        frr = 1 - tpr

        # ROC曲线 - 修改为FAR-FRR曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, frr, color='darkorange', lw=2, label=f'FAR-FRR curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(eer, eer, color='red', label=f'EER = {eer:.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FAR')
        plt.ylabel('FRR')
        plt.title('FAR-FRR curve')
        plt.legend(loc="upper right")

        # 相似度分布直方图
        plt.subplot(1, 2, 2)
        pos_sims, neg_sims = self._get_last_similarities()
        plt.hist(pos_sims, bins=50, alpha=0.5, label='正样本对', color='blue')
        plt.hist(neg_sims, bins=50, alpha=0.5, label='负样本对', color='red')
        plt.xlabel('余弦相似度')
        plt.ylabel('频数')
        plt.title('正负样本对相似度分布')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()




    def _get_last_similarities(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取最后一次计算的正负样本对相似度"""
        if not hasattr(self, 'last_pos_sims') or not hasattr(self, 'last_neg_sims'):
            raise ValueError("尚未计算相似度，请先调用compute_similarities方法")
        return self.last_pos_sims, self.last_neg_sims


if __name__ == "__main__":
    # 配置参数
    DATASET_PATH = "../dataset/CASIA"  # 替换为实际路径
    WEIGHT_PATH = "../weights/model_swint56.pth"  # 替换为实际路径
    CUSTOM_NUM_PAIRS = 2000  # None表示自动计算，或指定自定义数量

    # 初始化处理器
    processor = CASIADatasetProcessor(DATASET_PATH, WEIGHT_PATH)

    # 构建样本对 (可自定义数量)
    positive_pairs, negative_pairs = processor.build_pairs(CUSTOM_NUM_PAIRS)

    # 计算相似度
    pos_sims, neg_sims = processor.compute_similarities(positive_pairs, negative_pairs)

    # 评估性能并绘制曲线
    processor.evaluate_performance(pos_sims, neg_sims)