import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
from typing import Union, Optional
import torchvision.transforms as transforms


def extract_resnet18_features(
        image_input: Union[str, np.ndarray, Image.Image],  # 支持路径/numpy数组/PIL图像
        weight_path: str = r"C:\Users\CXY\Desktop\graduationDesign\project\palmVein\weights\resNet18-all.pth",
        roi_size: int = 224,
        device: Optional[str] = None,
) -> torch.Tensor:
    """
    从文件路径或内存图像中提取特征向量 (使用ResNet18)

    参数:
        image_input: 输入图像（支持三种形式）:
            - str: 图像文件路径
            - np.ndarray: 彩色图像数组 (H, W, 3)
            - PIL.Image: 彩色PIL图像对象
        weight_path: 模型权重路径
        roi_size: 输入到模型的图像大小 (默认224)
        device: 计算设备 (默认自动选择cuda/cpu)

    返回:
        torch.Tensor: 512维特征向量 (CPU tensor)
    """
    # --- 1. 初始化设备 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 初始化模型 (懒加载，避免重复初始化) ---
    if not hasattr(extract_resnet18_features, 'model'):
        # 加载ResNet18
        model = models.resnet18(pretrained=False)

        # 修改最后一层以匹配训练时的类别数（比如200类）
        original_fc_in_features = model.fc.in_features
        model.fc = nn.Linear(original_fc_in_features, 200)

        # 加载训练好的权重
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)  # 兼容直接保存和用字典保存的情况
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # 去掉分布式训练时的"module."前缀
        model.load_state_dict(state_dict, strict=True)

        # 替换最后的分类头，提取特征
        model.fc = nn.Identity()

        # 缓存模型
        model = model.to(device).eval()
        extract_resnet18_features.model = model
    else:
        model = extract_resnet18_features.model

    # --- 3. 图像预处理 ---
    transform = transforms.Compose([
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 处理不同输入类型
    if isinstance(image_input, str):
        # 文件路径
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3 or image_input.shape[2] != 3:
            raise ValueError(f"输入必须是形状为 (H, W, 3) 的彩色图像，实际形状: {image_input.shape}")
        img = Image.fromarray(image_input.astype('uint8'), 'RGB')
    elif isinstance(image_input, Image.Image):
        # PIL图像
        img = image_input.convert('RGB')
    else:
        raise TypeError("不支持的输入类型，应为路径/numpy数组/PIL图像")

    # --- 4. 提取特征 ---
    with torch.no_grad():
        input_tensor = transform(img).unsqueeze(0).to(device)
        features = model(input_tensor).squeeze(0)

    return features.cpu()
