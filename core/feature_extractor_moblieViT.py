import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Union, Optional
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from core.model_mobileViT import mobile_vit_xx_small
def apply_gray_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """函数式实现：RGB→灰度→CLAHE→复制三通道"""
    # PIL转OpenCV格式
    img_np = np.array(img)

    # RGB转灰度
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)

    # 复制为三通道并转回PIL格式
    rgb = np.stack([enhanced] * 3, axis=-1)  # HWC格式
    return Image.fromarray(rgb)

def extract_mobilevit_features(
        image_input: Union[str, np.ndarray, Image.Image],
        model_type: str = "mobile_vit_xx_small",
        weight_path: str= r"C:\Users\CXY\Desktop\graduationDesign\project\palmVein\weights\mobileViT\epoch_19_S2.pth",
        img_size: int = 224,
        device: Optional[str] = None,
        use_global_pool: bool = True
) -> torch.Tensor:
    """
    使用MobileViT提取图像特征向量

    参数:
        image_input: 输入图像 (文件路径/numpy数组/PIL图像)
        model_type: MobileViT模型类型 (支持"mobile_vit_xx_small"/"mobile_vit_x_small"/"mobile_vit_small")
        weight_path: 自定义权重路径 (None则使用默认架构)
        img_size: 输入图像大小
        device: 计算设备 (默认自动选择)
        use_global_pool: 是否使用全局平均池化作为特征 (False则使用最后的卷积特征图)

    返回:
        torch.Tensor: 特征向量 (维度取决于模型)
    """
    # --- 设备初始化 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- 模型初始化 (惰性加载) ---
    if not hasattr(extract_mobilevit_features, 'model'):
        # 1. 创建模型
        model = globals()[model_type](num_classes=0)  # num_classes=0表示移除分类头

        # 2. 加载权重 (如果提供)
        if weight_path:
            state_dict = torch.load(weight_path, map_location='cpu')
            if "model" in state_dict:  # 处理checkpoint格式
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True)

        # 3. 修改模型输出
        if use_global_pool:
            # 使用全局平均池化
            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:
            # 移除分类器，直接返回最后的卷积特征
            model.classifier = nn.Identity()

        model = model.to(device).eval()
        extract_mobilevit_features.model = model
    else:
        model = extract_mobilevit_features.model


    # --- 图像预处理 ---
    transform = transforms.Compose([
        transforms.Lambda(lambda x: apply_gray_clahe(x, clip_limit=2.0, grid_size=(8,8))),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 输入类型处理
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # 灰度图转RGB
            image_input = np.stack([image_input] * 3, axis=-1)
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise TypeError("不支持的输入类型")

    # --- 特征提取 ---
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0).to(device)
        features = model(img_tensor).squeeze(0)

    return features.cpu()


# 示例用法
if __name__ == "__main__":
    # 使用默认small模型 (无预训练权重)
    features = extract_mobilevit_features("example.jpg")
    print(f"特征维度: {features.shape}")

    # 使用自定义权重 (假设是分类模型)
    features = extract_mobilevit_features(
        "example.jpg",
        model_type="mobile_vit_small",
        weight_path="path/to/mobilevit_s.pt",
        use_global_pool=False
    )
