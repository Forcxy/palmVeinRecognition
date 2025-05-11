import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Union, Optional
from core.model_swinT_triplet import swin_tiny_patch4_window7_224
import cv2


def extract_swin_triplet_features(
        image_input: Union[str, np.ndarray, Image.Image],  # 支持路径/numpy数组/PIL图像
        weight_path: str = r"C:\Users\CXY\Desktop\graduationDesign\project\palmVein\weights\swinT-triplet\R-pre-M4\checkpoint_11.pth",
        roi_size: int = 224,
        device: Optional[str] = None,
        embedding_size: int = 512
) -> torch.Tensor:
    """
    使用Swin Transformer + Triplet Loss模型提取特征向量

    参数:
        image_input: 输入图像（支持三种形式）:
            - str: 图像文件路径
            - np.ndarray: 灰度图像数组 (H, W) 或 (H, W, 1)
            - PIL.Image: 灰度/彩色PIL图像对象
        weight_path: 模型权重路径
        roi_size: 输入图像大小
        device: 计算设备 (默认自动选择cuda/cpu)
        embedding_size: 特征向量维度

    返回:
        torch.Tensor: 特征向量 (CPU tensor)
    """
    # --- 1. 初始化设备 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 初始化模型 (惰性加载，避免重复初始化) ---
    if not hasattr(extract_swin_triplet_features, 'model'):
        # 创建模型 (使用文档1中的SwinTransformer结构)
        model = swin_tiny_patch4_window7_224(embedding_size=embedding_size)

        # 加载权重 (使用文档7中的权重加载逻辑)
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 过滤不匹配的键
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)

        model.load_state_dict(model_dict, strict=False)
        model = model.to(device).eval()

        # 缓存模型
        extract_swin_triplet_features.model = model
    else:
        model = extract_swin_triplet_features.model

    # --- 3. 图像预处理 (使用文档5中的预处理流程) ---
    def apply_gray_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
        """将RGB图像转换为灰度并应用CLAHE增强"""
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(gray)
        rgb = np.stack([enhanced] * 3, axis=-1)
        return Image.fromarray(rgb)

    transform = transforms.Compose([
        transforms.Lambda(apply_gray_clahe),  # 灰度+CLAHE增强
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 处理不同输入类型
    if isinstance(image_input, str):
        # 情况1: 文件路径
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        # 情况2: numpy数组
        if image_input.ndim == 2:
            img = Image.fromarray(image_input).convert('RGB')
        else:
            img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        # 情况3: PIL图像
        img = image_input.convert('RGB')
    else:
        raise TypeError("不支持的输入类型，应为路径/numpy数组/PIL图像")

    # --- 4. 推理 ---
    with torch.no_grad():
        input_tensor = transform(img).unsqueeze(0).to(device)
        features = model(input_tensor).squeeze(0)

    return features.cpu()


if __name__ == '__main__':
    img_path = r"C:\Users\CXY\Desktop\graduationDesign\src\palmVeinRecognition\swin_triple\polyu_0001_1.jpg"
    feat = extract_swin_triplet_features(img_path)
    print(f"特征向量形状: {feat.shape}")
    print(f"前5个特征值: {feat[:5].numpy()}")