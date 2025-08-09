import cv2
import numpy as np
import argparse
import os

import cv2
import numpy as np

def generate_binary_mask_01(original_image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    生成一个值为 0 和 1 的二值掩码 (binary mask)，适用于白色背景下的物体分割。
    无连通域分析，直接返回整体前景区域。
    """
    if verbose:
        print("--- [generate_binary_mask_01] 使用HSV分割白色背景物体（无连通域筛选） ---")

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose:
            print("错误: 输入图像必须是3通道BGR格式。")
        return None

    height, width = original_image.shape[:2]

    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # 背景为白色区域: 饱和度S低 + 亮度V高
    white_bg_mask = cv2.inRange(hsv, (0, 0, 240), (180, 10, 255))
    fg_mask = cv2.bitwise_not(white_bg_mask)  # 此时 fg_mask 是 0/255 的二值图

    if verbose:
        white_ratio = np.mean(white_bg_mask > 0)
        print(f"[HSV] 背景白色比例: {white_ratio*100:.2f}%")

    # 边缘增强（连接断裂区域）
    edges = cv2.Canny(fg_mask, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

    # 形态学处理（去除小噪声，填补小孔洞）
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if verbose:
        print(f"形态学处理完成，核大小: {kernel_size}")

    # 直接返回完整前景掩码（不做最大连通域筛选）
    final_mask_01 = (fg_mask / 255).astype(np.uint8)

    if verbose:
        print("生成 0/1 二值掩码完成（未使用连通域筛选）。")

    return final_mask_01


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 HSV 方法对白色背景图像进行主体分割")
    parser.add_argument("--image_path", type=str, default="/root/LucidDreamer/guidance/test_imgs/AAAI.png", help="输入图像路径")
    parser.add_argument("--save_path", type=str, default="/root/LucidDreamer/guidance/test_imgs/mask_binary.png", help="保存可视化掩码的路径")
    parser.add_argument("--verbose", action="store_true", help="打印详细调试信息")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"错误：找不到文件 {args.image_path}")
        exit(1)

    image = cv2.imread(args.image_path)
    if image is None:
        print("错误：图像读取失败。")
        exit(1)

    # 调用新函数，得到 0/1 掩码
    mask_01 = generate_binary_mask_01(image, verbose=args.verbose)

    if mask_01 is not None:
        print(f"生成的掩码的唯一值: {np.unique(mask_01)}") # 应该会打印 [0 1]
        print(f"掩码的数据类型: {mask_01.dtype}")

        # 在这里，你可以使用 mask_01 进行任何计算
        # 例如: masked_image = image * mask_01[:, :, np.newaxis]

        # --- 为了保存和可视化，创建一个 0/255 版本的掩码 ---
        mask_to_save = (mask_01 * 255).astype(np.uint8)

        save_path = args.save_path
        if save_path is None:
            base, ext = os.path.splitext(args.image_path)
            save_path = base + "_mask_binary.png"

        cv2.imwrite(save_path, mask_to_save)
        print(f"[完成] 已保存可视化掩码 (0/255) 至: {save_path}")
    else:
        print("[失败] 蒙版生成失败。")