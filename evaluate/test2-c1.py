import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from typing import List, Tuple, Dict
from core.feature_extractor_C1 import extract_swin_features
import matplotlib.pyplot as plt
import cv2
from core.image_processor import enhance_image
from core.feature_normalization import l2_normalize

# 全局matplotlib设置
plt.rcParams['font.family'] = ['Times New Roman']  # 设置字体族
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class CASIADatasetProcessor:
    """
    CASIA掌静脉数据集处理器，主要功能包括：
    - 使用Swin Transformer提取特征
    - 生成正负样本对
    - 评估识别性能（CRR、ROC、EER等）
    - 可视化结果
    """

    def __init__(self, dataset_path: str, weight_path: str, device: str = None):
        """
        初始化数据集处理器
        
        参数:
            dataset_path: 数据集目录路径
            weight_path: 预训练模型权重路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.dataset_path = dataset_path
        self.weight_path = weight_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = sorted(os.listdir(dataset_path))  # 获取所有类别名
        self.features_cache = {}  # 特征缓存字典
        self._validate_dataset()  # 验证数据集

    # ====================== 数据集验证 ======================
    def _validate_dataset(self):
        """验证数据集结构和内容是否符合要求"""
        print("正在验证数据集结构...")
        for cls in self.classes:
            cls_path = os.path.join(self.dataset_path, cls)
            if not os.path.isdir(cls_path):
                raise ValueError(f"无效的类别路径: {cls_path}")

            images = os.listdir(cls_path)
            if len(images) < 2:
                raise ValueError(f"类别 {cls} 中的图像数量不足 (至少需要2张)")
        print(f"数据集验证通过，共发现 {len(self.classes)} 个类别")

    # ====================== 特征提取 ======================
    def extract_all_features(self, force_reload: bool = False):
        """
        提取数据集中所有图像的特征
        
        参数:
            force_reload: 是否强制重新提取特征（忽略缓存）
        """
        if not force_reload and self.features_cache:
            print("使用缓存的特征")
            return

        print("开始提取所有图像的特征...")
        self.features_cache = {}

        for cls in tqdm(self.classes, desc="提取特征进度"):
            cls_path = os.path.join(self.dataset_path, cls)  # 表示类别文件路径
            images = sorted(os.listdir(cls_path))  # 获取类别文件夹下的所有图片文件名
            cls_features = []

            for img_name in images:
                try:
                    img_path = os.path.join(cls_path, img_name)

                    # 1. 读取原始图像
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告：无法读取图像 {img_path}")
                        continue

                    # 2. 图像增强处理,均转成灰度图像
                    enhanced_img = enhance_image(img)

                    # 3. 使用Swin Transformer提取特征
                    feature = extract_swin_features(
                        image_input=enhanced_img,
                        weight_path=self.weight_path,
                        device=self.device
                    )
                    #feature = l2_normalize(feature)
                    cls_features.append(feature)

                except Exception as e:
                    print(f"处理图像 {img_name} 时出错: {str(e)}")
                    continue

            if cls_features:  # 确保特征列表不为空
                # 示例 { “Raina”: 10 * 特征向量}
                self.features_cache[cls] = torch.stack(cls_features)
            else:
                print(f"警告：类别 {cls} 没有成功提取任何特征")

        print("特征提取完成")

    # ====================== 样本对生成 ======================
    def build_pairs(self, custom_num_pairs: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        生成正负样本对用于评估
        使用类型注解，更容易维护和查找错误
        参数:
            custom_num_pairs: 自定义生成的对数（None表示自动计算）
            
        返回:
            positive_pairs: 正样本对列表 [(类别索引, 图像索引1, 图像索引2), ...]
            negative_pairs: 负样本对列表 [(类别索引1, 图像索引1, 类别索引2, 图像索引2), ...]
        """
        self.extract_all_features()

        # 计算默认的对数
        n_classes = len(self.classes)  # 类别总数
        m_images = len(next(iter(self.features_cache.values())))  # 获取首个类别的图片数量

        if custom_num_pairs is None:
            num_pairs = n_classes * m_images * (m_images - 1) // 2
        else:
            num_pairs = custom_num_pairs

        print(f"\n生成样本对 (目标数量: {num_pairs})")
        print(f"类别数: {n_classes}, 每类图像数: {m_images}")

        # 生成正样本对（同一类别内的所有可能组合）
        positive_pairs = []
        for cls_idx, cls in enumerate(self.classes):  # enumerate返回 索引, 值
            img_indices = list(range(m_images))  # img_indices 是类别索引列表
            for img1, img2 in combinations(img_indices, 2):
                positive_pairs.append((cls_idx, img1, img2))

        negative_pairs = set()
        cls_pairs = list(combinations(range(n_classes), 2))  # cls1 < cls2 组合去重

        while len(negative_pairs) < num_pairs:
            cls1_idx, cls2_idx = random.choice(cls_pairs)
            img1_idx = random.randint(0, m_images - 1)
            img2_idx = random.randint(0, m_images - 1)
            negative_pairs.add((cls1_idx, img1_idx, cls2_idx, img2_idx))  # 自动去重

        # 生成负样本对（随机选择不同类别的图像对）
        negative_pairs = list(negative_pairs)
        # for _ in range(num_pairs):
        #     cls1_idx, cls2_idx = random.sample(range(n_classes), 2)
        #     cls1, cls2 = self.classes[cls1_idx], self.classes[cls2_idx]
        #     img1_idx = random.randint(0, m_images - 1)
        #     img2_idx = random.randint(0, m_images - 1)
        #     negative_pairs.append((cls1_idx, img1_idx, cls2_idx, img2_idx))

        # 如果指定了自定义数量且小于自动计算的正样本对数，则进行随机采样
        if custom_num_pairs is not None and custom_num_pairs < len(positive_pairs):
            positive_pairs = random.sample(positive_pairs, custom_num_pairs)
            negative_pairs = negative_pairs[:custom_num_pairs]

        print(f"正样本对数量: {len(positive_pairs)}")
        print(f"负样本对数量: {len(negative_pairs)}")

        return positive_pairs, negative_pairs

    # ====================== 相似度计算 ======================
    def compute_similarities(
            self,
            positive_pairs: List,
            negative_pairs: List,
            metric: str = "euclidean"  # 新增参数，可选 "cosine" 或 "euclidean"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算正负样本对的相似度（支持余弦/欧氏距离）

        参数:
            positive_pairs: 正样本对列表
            negative_pairs: 负样本对列表
            metric: 距离度量方式，可选 "cosine" 或 "euclidean"

        返回:
            pos_sims: 正样本对的相似度数组
            neg_sims: 负样本对的相似度数组
        """
        print(f"\n计算{metric}相似度...")

        def _compute(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
            """根据metric参数计算相似度"""
            if metric == "cosine":
                return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
            elif metric == "euclidean":
                dist = F.pairwise_distance(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
                return 1 / (1 + dist)  # 将欧氏距离转换为相似度（0~1范围）
            else:
                raise ValueError(f"不支持的度量方式: {metric}")

        # 计算正样本对相似度
        pos_sims = []
        for cls_idx, img1_idx, img2_idx in tqdm(positive_pairs, desc="正样本对处理"):
            cls = self.classes[cls_idx]
            feat1 = self.features_cache[cls][img1_idx]
            feat2 = self.features_cache[cls][img2_idx]
            pos_sims.append(_compute(feat1, feat2))

        # 计算负样本对相似度
        neg_sims = []
        for cls1_idx, img1_idx, cls2_idx, img2_idx in tqdm(negative_pairs, desc="负样本对处理"):
            cls1 = self.classes[cls1_idx]
            cls2 = self.classes[cls2_idx]
            feat1 = self.features_cache[cls1][img1_idx]
            feat2 = self.features_cache[cls2][img2_idx]
            neg_sims.append(_compute(feat1, feat2))

        # 缓存结果供后续使用
        self.last_pos_sims = np.array(pos_sims)
        self.last_neg_sims = np.array(neg_sims)

        return self.last_pos_sims, self.last_neg_sims

    # ====================== 性能评估 ======================
    def evaluate_performance(self, pos_sims: np.ndarray, neg_sims: np.ndarray, plot: bool = True):
        """
        评估识别性能指标，包括ROC、AUC和EER
        
        参数:
            pos_sims: 正样本对的相似度数组
            neg_sims: 负样本对的相似度数组
            plot: 是否生成性能曲线图
        """
        print("\n正在评估性能...")
        print(f"正样本对数量: {len(pos_sims)}")
        print(f"负样本对数量: {len(neg_sims)}")

        # 准备标签和预测分数
        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_score = np.concatenate([pos_sims, neg_sims])

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 计算FRR（错误拒绝率）
        frr = 1 - tpr

        # 计算EER（等错误率）
        eer_idx = np.nanargmin(np.absolute(frr - fpr))
        eer = frr[eer_idx]
        eer_threshold = thresholds[eer_idx]

        print(f"ROC曲线下面积 (AUC): {roc_auc:.4f}")
        print(f"等错误率 (EER): {eer:.4f} (阈值={eer_threshold:.4f})")

        if plot:
            self._plot_curves(fpr, tpr, roc_auc, eer)

        self.last_threshold = eer_threshold

    def evaluate_threshold_crr(self, pos_sims: np.ndarray, neg_sims: np.ndarray, threshold: float):
        """
        使用指定阈值评估识别准确率
        
        参数:
            pos_sims: 正样本对的相似度数组
            neg_sims: 负样本对的相似度数组
            threshold: 判定阈值
        """
        print(f"\n使用阈值 {threshold:.4f} 评估识别准确率...")

        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_pred = np.concatenate([pos_sims, neg_sims]) >= threshold

        correct = np.sum(y_pred == y_true)
        total = len(y_true)
        acc = correct / total

        print(f"基于阈值的识别准确率 (CRR): {acc:.4f} ({correct}/{total})")

    def evaluate_crr_top1(self):
        """评估Top-1正确识别率"""
        print("\n正在评估Top-1识别率...")
        self.extract_all_features()

        total = 0
        correct = 0

        for cls_idx, cls in enumerate(self.classes):
            cls_features = self.features_cache[cls]

            for i, probe_feat in enumerate(cls_features):
                max_sim = -1
                predicted_class = None

                for candidate_cls_idx, candidate_cls in enumerate(self.classes):
                    candidate_feats = self.features_cache[candidate_cls]
                    sims = F.cosine_similarity(probe_feat.unsqueeze(0), candidate_feats).cpu().numpy()
                    mean_sim = np.max(sims)

                    if mean_sim > max_sim:
                        max_sim = mean_sim
                        predicted_class = candidate_cls_idx

                total += 1
                if predicted_class == cls_idx:
                    correct += 1

        acc = correct / total
        print(f"Top-1识别率 (CRR): {acc:.4f} ({correct}/{total})")

    # ====================== 可视化 ======================
    def _plot_curves(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, eer: float):
        """
        绘制FAR-FRR曲线（原始配色单图版）

        参数:
            fpr: 假阳性率数组 (FAR)
            tpr: 真阳性率数组
            roc_auc: ROC曲线下面积
            eer: 等错误率
        """
        plt.figure(figsize=(8, 6))  # 保持专业单图尺寸

        # 计算FRR（错误拒绝率）
        frr = 1 - tpr

        # ---------- 绘制FAR-FRR曲线 ----------
        plt.plot(fpr, frr, color='darkorange', lw=2,
                 label=f'FAR-FRR Curve (AUC = {roc_auc:.2f})')

        # 绘制参考线和EER点（原始配色）
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.scatter(eer, eer, color='red', s=80, label=f'EER = {eer:.3f}')

        # 坐标轴设置
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('FAR')
        plt.ylabel('FRR')
        plt.title('FAR vs FRR Curve')

        # 图例和网格
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ====================== 工具方法 ======================
    def _get_last_threshold(self) -> float:
        """获取上次性能评估计算得到的最佳阈值"""
        if not hasattr(self, 'last_threshold'):
            raise ValueError("没有可用的阈值 - 请先运行evaluate_performance方法")
        return self.last_threshold

    def _get_last_similarities(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取上次计算的相似度结果"""
        if not hasattr(self, 'last_pos_sims') or not hasattr(self, 'last_neg_sims'):
            raise ValueError("没有可用的相似度数据 - 请先运行compute_similarities方法")
        return self.last_pos_sims, self.last_neg_sims


if __name__ == "__main__":
    # 配置参数
    DATASET_PATH = r"C:\Users\CXY\Desktop\graduationDesign\project\palmVein\dataset\CASIA"  # 数据集路径
    #DATASET_PATH = r"C:\Users\CXY\Desktop\graduationDesign\dataset\AllVeinDataset\HFUT_split\val"  # 数据集路径
    WEIGHT_PATH = "../weights/model_swint63-C1-E.pth"  # 模型权重路径
    CUSTOM_NUM_PAIRS = None # 自定义样本对数（None表示自动计算）

    # 初始化处理器
    processor = CASIADatasetProcessor(DATASET_PATH, WEIGHT_PATH)

    # 生成样本对
    positive_pairs, negative_pairs = processor.build_pairs(CUSTOM_NUM_PAIRS)

    # 计算相似度
    pos_sims, neg_sims = processor.compute_similarities(positive_pairs, negative_pairs)

    # 评估性能
    processor.evaluate_performance(pos_sims, neg_sims)

    # 使用最佳阈值评估识别率
    best_threshold = processor._get_last_threshold()
    processor.evaluate_threshold_crr(pos_sims, neg_sims, best_threshold)
