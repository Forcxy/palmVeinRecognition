import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Union, Optional
from core.model import swin_tiny_patch4_window7_224


def extract_swin_features(
        image_input: Union[str, np.ndarray, Image.Image],  # 支持路径/numpy数组/PIL图像
        weight_path: str,
        roi_size: int = 224,
        device: Optional[str] = None,
        input_channels: int = 3  # 新增参数：输入通道数 (1或3)
) -> torch.Tensor:
    """
    从文件路径或内存图像中提取特征向量

    参数:
        image_input: 输入图像（支持三种形式）:
            - str: 图像文件路径
            - np.ndarray: 灰度图像数组 (H, W) 或 (H, W, 1)
            - PIL.Image: 灰度/彩色PIL图像对象
        weight_path: 模型权重路径
        roi_size: 输入图像大小
        device: 计算设备 (默认自动选择cuda/cpu)
        input_channels: 输入通道数 (1表示灰度，3表示彩色，默认为3)

    返回:
        torch.Tensor: 768维特征向量 (CPU tensor)
    """
    # 验证输入通道数
    if input_channels not in [1, 3]:
        raise ValueError("input_channels must be either 1 or 3")

    # --- 1. 初始化设备 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 初始化模型 (惰性加载，避免重复初始化) ---
    # 使用函数名+输入通道数作为缓存键，以支持不同通道数的模型
    cache_key = f"{extract_swin_features.__name__}_ch{input_channels}"

    if not hasattr(extract_swin_features, cache_key):
        model = swin_tiny_patch4_window7_224(num_classes=200, pretrained=False)

        # 加载权重
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        # 修改forward方法
        def forward_features(self, x):
            x, H, W = self.patch_embed(x)
            x = self.pos_drop(x)
            for layer in self.layers:
                x, H, W = layer(x, H, W)
            x = self.norm(x)
            return x.mean(dim=1)

        model.forward = forward_features.__get__(model)
        model = model.to(device).eval()
        setattr(extract_swin_features, cache_key, model)  # 缓存模型
    else:
        model = getattr(extract_swin_features, cache_key)

    # --- 3. 图像预处理 ---
    # 根据输入通道数调整归一化参数
    if input_channels == 1:
        norm_mean = [0.5]
        norm_std = [0.5]
    else:
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # 处理不同输入类型
    if isinstance(image_input, str):
        # 情况1: 文件路径
        img = Image.open(image_input)
        if input_channels == 1:
            img = img.convert('L')
    elif isinstance(image_input, np.ndarray):
        # 情况2: numpy数组
        if len(image_input.shape) == 3 and image_input.shape[2] == 1:
            image_input = image_input.squeeze(2)
        img = Image.fromarray(image_input.astype('uint8'))
        if input_channels == 1 and img.mode != 'L':
            img = img.convert('L')
    elif isinstance(image_input, Image.Image):
        # 情况3: PIL图像
        img = image_input
        if input_channels == 1 and img.mode != 'L':
            img = img.convert('L')
    else:
        raise TypeError("不支持的输入类型，应为路径/numpy数组/PIL图像")

    # 通道数处理
    if input_channels == 1:
        # 单通道输入不需要转换为RGB
        pass
    else:
        # 三通道输入：确保是RGB格式
        if img.mode == 'L':
            img = Image.merge('RGB', (img, img, img))
        elif img.mode != 'RGB':
            img = img.convert('RGB')

    # --- 4. 推理 ---
    with torch.no_grad():
        input_tensor = transform(img).unsqueeze(0).to(device)
        features = model(input_tensor).squeeze(0)

    return features.cpu()