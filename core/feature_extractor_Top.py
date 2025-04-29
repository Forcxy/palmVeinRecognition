from typing import Union, Optional, Literal
import torch
import numpy as np
from PIL import Image
from core.feature_extractor_resnet import extract_resnet18_features
from core.feature_extractor_swint import extract_swin_features
from core.feature_extractor_viT import extract_vit_features
from core.feature_extractor_moblieViT import extract_mobilevit_features

def extract_features(
        model_type: Literal['resnet', 'swin', 'viT', 'mobileViT'],
        image_input: Union[str, np.ndarray, Image.Image],
        weight_path: str = None,  # 设为可选
        roi_size: int = 224,
        device: Optional[str] = None,
        **kwargs
) -> torch.Tensor:
    """
    统一特征提取接口，支持不同模型架构

    参数:
        model_type: 模型类型 ('resnet' 或 'swin')
        image_input: 输入图像（路径/numpy数组/PIL图像）
        weight_path: 模型权重路径
        roi_size: 输入图像大小
        device: 计算设备 (默认自动选择)
        **kwargs: 模型特定参数:
            - 对于resnet: use_clahe (bool)
            - 对于swin: 无额外参数

    返回:
        torch.Tensor: 特征向量 (CPU tensor)
    """
    # 注意不能传None
    if model_type == 'resnet':
        return extract_resnet18_features(
            image_input=image_input,
            roi_size=roi_size,
            device=device,
        )

    elif model_type == 'swin':
        return extract_swin_features(
            image_input=image_input,
            roi_size=roi_size,
            device=device
        )
    elif model_type == 'viT':
        return extract_vit_features(
            image_input=image_input,
            weight_path=weight_path,
            img_size=roi_size,
            device=device,
            use_cls_token=True
        )
    elif model_type == 'mobileViT':
        return extract_mobilevit_features(
            image_input=image_input,
            weight_path=weight_path,
            img_size=roi_size,
            device=device
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 请选择 'resnet' 或 'swin'或")

# 使用示例:
# 1. 使用ResNet提取特征
# features = extract_features(
#     model_type='resnet',
#     image_input='path/to/image.jpg',
#     weight_path='resnet_weights.pth',
#     use_clahe=True
# )

# 2. 使用Swin Transformer提取特征
# features = extract_features(
#     model_type='swin',
#     image_input='path/to/image.jpg',
#     weight_path='swin_weights.pth'
# )