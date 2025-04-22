import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Union, Optional
from core.model import swin_tiny_patch4_window7_224

def extract_swin_features(
        image_input: Union[str, np.ndarray, Image.Image],  # 支持路径/numpy数组/PIL图像
        weight_path: str = r"C:\Users\CXY\Desktop\graduationDesign\project\palmVein\weights\model_swint51-C3.pth",
        roi_size: int = 224,
        device: Optional[str] = None
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

    返回:
        torch.Tensor: 768维特征向量 (CPU tensor)
    """
    # --- 1. 初始化设备 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 初始化模型 (惰性加载，避免重复初始化) ---
    if not hasattr(extract_swin_features, 'model'):
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
        extract_swin_features.model = model  # 缓存模型
    else:
        model = extract_swin_features.model

    # --- 3. 图像预处理 ---
    transform = transforms.Compose([
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 处理不同输入类型
    if isinstance(image_input, str):
        # 情况1: 文件路径
        img = Image.open(image_input).convert('L')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3:
            print("输入图像不是三通道")
        img = Image.fromarray(image_input.astype('uint8'), 'RGB')
    elif isinstance(image_input, Image.Image):
        # 情况3: PIL图像
        img = image_input
    else:
        raise TypeError("不支持的输入类型，应为路径/numpy数组/PIL图像")


    # --- 4. 推理 ---
    with torch.no_grad():
        input_tensor = transform(img).unsqueeze(0).to(device)
        features = model(input_tensor).squeeze(0)

    return features.cpu()

