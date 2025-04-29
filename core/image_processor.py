
import cv2
import numpy as np
def anisotropic_diffusion(img, iterations=10, kappa=30, gamma=0.1):
    """
    Perona-Malik各向异性扩散去噪
    参数：
        img: 输入图像(灰度)
        iterations: 迭代次数
        kappa: 边缘敏感阈值
        gamma: 时间步长
    """
    img = img.astype(np.float32)

    for _ in range(iterations):
        # 计算梯度
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

        # 计算梯度幅值
        grad_magnitude = np.square(grad_x) + np.square(grad_y)

        # 计算传导系数
        flow = np.exp(-grad_magnitude / (kappa ** 2))

        # 计算扩散量
        diffusion = cv2.Laplacian(img, cv2.CV_32F)

        # 更新图像
        img += gamma * (diffusion * flow)

    # 确保结果在0-255范围内
    return np.clip(img, 0, 255).astype(np.uint8)

# ---------------- 图像处理模块 ----------------

def enhance_image(img,mode=None,clip_limit=2.0, grid_size=8):
    # 若是 RGB 图像，先转为灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #尺度归一化
    # resized = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)

    # 2. 全局直方图均衡化
    #equalized = cv2.equalizeHist(resized)

    # 3. 各向异性扩散去噪
    #denoised = anisotropic_diffusion(resized)

    # 4. 局部灰度归一化 (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )

    final = clahe.apply(img)  #暂时跳过去噪

    # 界面展示灰度图片
    if mode == "c1":
        return final
    # todo 图像增强修改返回通道数
    final_rgb = cv2.merge([final, final, final])  # 三通道复制
    # 返回numpy数组
    return final_rgb