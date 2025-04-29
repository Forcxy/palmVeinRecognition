import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Union, Optional
import torchvision.transforms as transforms
import cv2
from core.model_vit import vit_base_patch16_224

def extract_vit_features(
        image_input: Union[str, np.ndarray, Image.Image],
        model_type: str = "vit_base_patch16_224",
        weight_path: str = r"C:\Users\CXY\Desktop\graduationDesign\project\palmVein\weights\vit_half_2.pth",
        img_size: int = 224,
        device: Optional[str] = None,
        use_cls_token: bool = True
) -> torch.Tensor:

    """
    使用Vision Transformer提取图像特征向量
    
    参数:
        image_input: 输入图像 (文件路径/numpy数组/PIL图像)
        model_type: ViT模型类型 (默认"vit_base_patch16_224")
        weight_path: 自定义权重路径 (None则使用默认架构)
        img_size: 输入图像大小
        device: 计算设备 (默认自动选择)
        use_cls_token: 是否使用[CLS] token作为特征 (False则使用全局平均)
        
    返回:
        torch.Tensor: 特征向量 (维度取决于模型)
    """

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

    # --- 设备初始化 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    
    # --- 模型初始化 (惰性加载) ---
    if not hasattr(extract_vit_features, 'model'):
        # 1. 创建模型
        model = globals()[model_type](num_classes=0)  # num_classes=0表示移除分类头
        
        # 2. 加载权重 (如果提供)
        if weight_path:
            state_dict = torch.load(weight_path, map_location='cpu')
            if "model" in state_dict:  # 处理checkpoint格式
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True)
        
        # 3. 修改模型输出
        if use_cls_token:
            # 保持原样，使用[CLS] token
            model.head = nn.Identity()
        else:
            # 替换为全局平均池化
            model.head = nn.Sequential(
                nn.LayerNorm(model.embed_dim),
                nn.Linear(model.embed_dim, model.embed_dim),
                nn.Tanh()
            )
        
        model = model.to(device).eval()
        extract_vit_features.model = model
    else:
        model = extract_vit_features.model
    
    # --- 图像预处理 ---
    transform = transforms.Compose([
        transforms.Lambda(lambda x: apply_gray_clahe(x, clip_limit=2.0, grid_size=(8, 8))),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 输入类型处理
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # 灰度图转RGB
            image_input = np.stack([image_input]*3, axis=-1)
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise TypeError("不支持的输入类型")
    
    # --- 特征提取 ---
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        if use_cls_token:
            # 直接获取[CLS] token对应的特征
            features = model(img_tensor)  # [1, embed_dim]
        else:
            # 获取所有patch tokens并全局平均
            x = model.patch_embed(img_tensor)  # [1, num_patches, embed_dim]
            x = torch.cat([model.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = model.pos_drop(x + model.pos_embed)
            x = model.blocks(x)
            x = model.norm(x)
            features = x[:, 1:].mean(dim=1)  # 排除[CLS] token后平均
        
    return features.squeeze(0).cpu()  # 返回一维向量


# 示例用法
if __name__ == "__main__":
    # 使用默认模型 (无预训练权重)
    features = extract_vit_features("example.jpg")
    print(f"特征维度: {features.shape}")
    
    # 使用自定义权重 (假设是分类模型)
    features = extract_vit_features(
        "example.jpg",
        model_type="vit_base_patch16_224",
        weight_path="path/to/vit_base_patch16_224_in21k.pth",
        use_cls_token=False
    )