# palm_utils.py
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

mp_hands = mp.solutions.hands

# 初始化MediaPipe Hands（只需要初始化一次）
_hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_roi(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """提取ROI区域"""
    # MediaPipe 需要图像使用 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = _hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    keypoints_of_interest = {5: "Index MCP", 6: "Index PIP", 17: "Pinky MCP", 18: "Pinky PIP"}
    selected_keypoints = get_keypoints(image_rgb, results.multi_hand_landmarks[0], keypoints_of_interest)

    if len(selected_keypoints) < 4:
        print("[ERROR] 提取的关键点不足")
        return None, None

    try:
        detection_img = image_rgb.copy()
        square_points = calculate_square_points(
            selected_keypoints["Pinky MCP"],
            selected_keypoints["Index MCP"],
            selected_keypoints["Index PIP"]
        )
        draw_square(detection_img, square_points)
        # 额外绘制PIP关键点（小指PIP、食指PIP）
        pip_points = {
            "Pinky PIP": selected_keypoints["Pinky PIP"],
            "Index PIP": selected_keypoints["Index PIP"]
        }
        for point_name, point in pip_points.items():
            color = (0, 255, 0) if "PIP" in point_name else (0, 0, 255)
            cv2.circle(detection_img, point, 5, color, -1)
            # 连线：PIP点 → 对应MCP点
            mcp_point = selected_keypoints["Pinky MCP"] if "Pinky" in point_name else selected_keypoints["Index MCP"]
            cv2.line(detection_img, point, mcp_point, (255, 0, 0), 2)


        roi_img = crop_square_region(image, square_points)
        return detection_img, roi_img
    except Exception as e:
        print(f"[ERROR] ROI extraction failed: {str(e)}")
        return None, None

"""提取手部关键点坐标"""
def get_keypoints(image, hand_landmarks, keypoints_of_interest):
    """提取手部关键点坐标"""
    height, width, _ = image.shape
    selected_keypoints = {}
    for idx, landmark in enumerate(hand_landmarks.landmark):
        if idx in keypoints_of_interest:
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            selected_keypoints[keypoints_of_interest[idx]] = (cx, cy)
    return selected_keypoints
# 计算ROI的四个顶点
def calculate_square_points(pinky_mcp, index_mcp, index_pip):
    """计算ROI的四个顶点"""
    axis_vector = np.array(pinky_mcp) - np.array(index_mcp)
    vector1 = np.array(index_pip) - np.array(index_mcp)
    vector2 = np.array(index_mcp) - np.array(pinky_mcp)
    cross_product = np.cross(vector1, vector2)

    orthogonal_vector = np.array([-axis_vector[1], axis_vector[0]]) if cross_product < 0 else np.array([axis_vector[1], -axis_vector[0]])

    third_point = tuple((np.array(pinky_mcp) + orthogonal_vector).astype(int))
    fourth_point = tuple((np.array(index_mcp) + orthogonal_vector).astype(int))
    return pinky_mcp, index_mcp, third_point, fourth_point

def crop_square_region(image, points):
    """透视变换裁剪ROI"""
    src_points = np.array(points, dtype=np.float32)
    width = int(np.linalg.norm(src_points[0] - src_points[1]))
    dst_points = np.array([[0, 0], [width-1, 0], [0, width-1], [width-1, width-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (width, width))


def draw_square(image, points):
    point1, point2, point3, point4 = points

    # 绘制关键点（与第二个代码完全一致）
    cv2.circle(image, point1, 8, (0, 255, 0), -1)  # 绿点（小指MCP）
    cv2.circle(image, point2, 8, (0, 255, 0), -1)  # 绿点（食指MCP）
    cv2.circle(image, point3, 8, (0, 0, 255), -1)  # 红点（小指延伸点）
    cv2.circle(image, point4, 8, (0, 0, 255), -1)  # 红点（食指延伸点）

    # 绘制正方形边框（蓝色）
    cv2.line(image, point1, point2, (255, 0, 0), 3)  # 底边（小指MCP → 食指MCP）
    cv2.line(image, point1, point3, (255, 0, 0), 3)  # 左侧边（小指MCP → 小指延伸点）
    cv2.line(image, point2, point4, (255, 0, 0), 3)  # 右侧边（食指MCP → 食指延伸点）
    cv2.line(image, point3, point4, (255, 0, 0), 3)  # 顶边（小指延伸点 → 食指延伸点）