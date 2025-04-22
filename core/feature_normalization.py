import torch.nn.functional as F
import torch

def l2_normalize(features: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """L2归一化（支持批量处理）"""
    return F.normalize(features, p=2, dim=-1) if features.dim() > 1 else F.normalize(features.unsqueeze(0), p=2, dim=-1).squeeze(0)