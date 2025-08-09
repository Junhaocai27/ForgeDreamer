import cv2
import numpy as np
import os
import sys

def segment_object_on_solid_background(image_path: str):
    """
    从指定路径的图像中分割出纯色背景下的物体，并直接保存结果。
    优化版本，提供更好的细节处理。

    参数:
    image_path (str): 输入图像的文件路径。
    """
    # --- 步骤 1: 检查并读取图像 ---
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 '{image_path}'")
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误: OpenCV无法读取图像 '{image_path}'。请检查文件是否为有效的图像格式。")
        return
        
    print(f"成功读取图像: {os.path.basename(image_path)}")
    height, width = original_image.shape[:2]

    # --- 步骤 2: 多通道分析选择最佳通道 ---
    # 分析RGB各通道的方差，选择对比度最高的通道
    b, g, r = cv2.split(original_image)
    channels = [b, g, r]
    channel_names = ['Blue', 'Green', 'Red']
    
    # 计算每个通道的方差，选择方差最大的（对比度最高）
    variances = [np.var(channel) for channel in channels]
    best_channel_idx = np.argmax(variances)
    best_channel = channels[best_channel_idx]
    
    print(f"选择了{channel_names[best_channel_idx]}通道进行分割（方差: {variances[best_channel_idx]:.2f}）")

    # --- 步骤 3: 边缘检测辅助分割 ---
    # 使用Canny边缘检测来识别物体边界
    # 自动计算阈值：使用图像中位数的0.5倍和1.5倍
    median_val = np.median(best_channel)
    lower_thresh = max(0, int(0.5 * median_val))
    upper_thresh = min(255, int(1.5 * median_val))
    
    edges = cv2.Canny(best_channel, lower_thresh, upper_thresh)
    print(f"边缘检测阈值: {lower_thresh}-{upper_thresh}")

    # --- 步骤 4: 改进的Otsu二值化 ---
    # 对最佳通道应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(best_channel, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- 步骤 5: 结合边缘信息改进蒙版 ---
    # 使用边缘信息来增强物体边界
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(otsu_mask, dilated_edges)
    
    # --- 步骤 6: 智能形态学处理 ---
    # 根据图像大小自适应调整核大小
    kernel_size = max(3, min(9, int(min(height, width) / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 开运算去除小噪点
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 闭运算填充物体内部空洞
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    print(f"使用{kernel_size}x{kernel_size}椭圆核进行形态学处理...")

    # --- 步骤 7: 连通域分析保留最大区域 ---
    # 找到所有连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    if num_labels > 1:  # 除了背景外还有其他区域
        # 找到最大的连通域（除背景外）
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # 创建只包含最大连通域的蒙版
        final_mask = np.zeros_like(cleaned_mask)
        final_mask[labels == largest_component] = 255
        
        print(f"保留最大连通域，面积: {stats[largest_component, cv2.CC_STAT_AREA]} 像素")
    else:
        final_mask = cleaned_mask
        print("未发现有效的连通域")

    # --- 步骤 8: 边缘平滑处理 ---
    # 使用高斯模糊和阈值化来平滑边缘
    smoothed_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    _, final_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

    # --- 步骤 9: 创建羽化效果的Alpha通道 ---
    # 对蒙版应用轻微的高斯模糊来创建更自然的边缘
    alpha_channel = cv2.GaussianBlur(final_mask, (3, 3), 0)
    
    # 将原始图像转换为BGRA格式
    segmented_object_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    segmented_object_rgba[:, :, 3] = alpha_channel
    
    print("已应用羽化效果到Alpha通道...")

    # --- 步骤 10: 边界优化 ---
    # 检测边界区域并进行额外处理
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 创建更精确的蒙版
        refined_mask = np.zeros_like(final_mask)
        cv2.fillPoly(refined_mask, [largest_contour], 255)
        
        # 轻微膨胀以确保边缘完整
        refined_mask = cv2.dilate(refined_mask, np.ones((2, 2), np.uint8), iterations=1)
        
        # 更新Alpha通道
        segmented_object_rgba[:, :, 3] = refined_mask
        
        print("已优化物体边界...")

    # --- 步骤 11: 保存结果 ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path_segmented = f"{base_name}_segmented_optimized.png"
    output_path_mask = f"{base_name}_mask_optimized.png"

    try:
        # 保存带透明背景的分割物体
        cv2.imwrite(output_path_segmented, segmented_object_rgba)
        print(f"✅ 优化分割结果已保存至: {output_path_segmented}")

        # 保存最终蒙版
        cv2.imwrite(output_path_mask, segmented_object_rgba[:, :, 3])
        print(f"✅ 优化蒙版已保存至: {output_path_mask}")

        # 额外保存一个白色背景版本供查看
        white_bg_result = original_image.copy()
        mask_3channel = cv2.cvtColor(segmented_object_rgba[:, :, 3], cv2.COLOR_GRAY2BGR)
        mask_normalized = mask_3channel.astype(float) / 255.0
        
        # 创建白色背景
        white_background = np.ones_like(original_image) * 255
        white_bg_result = (original_image * mask_normalized + 
                          white_background * (1 - mask_normalized)).astype(np.uint8)
        
        output_path_white_bg = f"{base_name}_white_background.png"
        cv2.imwrite(output_path_white_bg, white_bg_result)
        print(f"✅ 白色背景版本已保存至: {output_path_white_bg}")

    except Exception as e:
        print(f"❌ 保存文件时发生错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    input_image_path = '/root/LucidDreamer/guidance/test_imgs/10.jpg'  # <--- 在这里修改你的图片路径！

    # 检查用户是否已修改默认路径
    if 'your_image.jpg' in input_image_path:
        print("-" * 60)
        print(">> 请先修改脚本中的 'input_image_path' 变量，使其指向您的图片文件。")
        print("-" * 60)
        sys.exit()

    print("--- 开始优化处理 ---")
    segment_object_on_solid_background(input_image_path)
    print("--- 处理完毕 ---")