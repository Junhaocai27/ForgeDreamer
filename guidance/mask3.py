import cv2
import numpy as np
import argparse
import os

def generate_advanced_mask_hsv(original_image: np.ndarray, verbose: bool = False) -> np.ndarray:
    if verbose:
        print("--- [generate_advanced_mask_hsv] 使用HSV分割白色背景物体 ---")

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        if verbose: print("错误: 输入图像必须是3通道BGR格式。")
        return None

    height, width = original_image.shape[:2]
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # 背景为白色区域: 饱和度S低 + 亮度V高
    white_bg_mask = cv2.inRange(hsv, (0, 0, 255), (180, 0, 255))
    fg_mask = cv2.bitwise_not(white_bg_mask)

    if verbose:
        white_ratio = np.mean(white_bg_mask > 0)
        print(f"[HSV] 背景白色比例: {white_ratio*100:.2f}%")

    # 加边缘强化
    edges = cv2.Canny(fg_mask, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg_mask = cv2.bitwise_or(fg_mask, dilated_edges)

    # 形态学处理
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 连通域提取
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels <= 1:
        if verbose: print("未找到主体区域，返回初步掩码。")
        return fg_mask

    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.zeros_like(fg_mask)
    final_mask[labels == largest_component] = 255
    if verbose: print(f"保留最大连通域作为主体，面积: {stats[largest_component, cv2.CC_STAT_AREA]}")

    # 轮廓平滑处理
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if verbose: print("未找到轮廓，直接返回。")
        return final_mask

    refined_mask = np.zeros_like(final_mask)
    cv2.drawContours(refined_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    refined_mask = cv2.dilate(refined_mask, np.ones((3, 3), np.uint8), iterations=1)
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)

    if verbose: print("边缘优化完成，生成最终掩码。")
    return refined_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 HSV 方法对白色背景图像进行主体分割")
    parser.add_argument("--image_path", type=str, default="/root/LucidDreamer/guidance/test_imgs/12.png", help="输入图像路径")
    parser.add_argument("--save_path", type=str, default="/root/LucidDreamer/guidance/test_imgs/mask3.jpg", help="可选，保存分割结果的路径")
    parser.add_argument("--verbose", action="store_true", help="打印详细调试信息")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"错误：找不到文件 {args.image_path}")
        exit(1)

    image = cv2.imread(args.image_path)
    if image is None:
        print("错误：图像读取失败。")
        exit(1)

    mask = generate_advanced_mask_hsv(image, verbose=args.verbose)

    if mask is not None:
        save_path = args.save_path
        if save_path is None:
            # 自动生成保存路径，保存到同目录并加后缀
            base, ext = os.path.splitext(args.image_path)
            save_path = base + "_mask.png"

        cv2.imwrite(save_path, mask)
        print(f"[完成] 已保存掩码至: {save_path}")
    else:
        print("[失败] 蒙版生成失败。")
