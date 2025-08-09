#!/bin/bash

# --- 配置 ---
# 设置 lora_add 命令的路径。如果它在你的 PATH 中，可以直接使用 "lora_add"。
# 否则，请提供完整路径，例如 LORA_ADD_CMD="python /path/to/your/lora_add_script.py"
LORA_ADD_CMD="lora_add"

# 合并时使用的默认 alpha 值 (权重)。对于简单平均合并，通常都设为 1.0。
ALPHA_1="1.0"
ALPHA_2="1.0"

# --- 输入 LoRA 文件列表 ---
# 在这里直接定义要合并的输入 LoRA 文件路径列表
# 每个文件路径用引号括起来，并用空格分隔
input_files=(
  "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/gasket_new_ex/up/step_inv_1000.safetensors"
  "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_rank16/step_inv_1000.safetensors"
  "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/gasket_new_ex/front/step_inv_1000.safetensors"
  "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/nut_new_ex/front_up_combined.safetensors"
  "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/nail_new_ex/step_inv_1000.safetensors"
#   "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/nut1/final_combine_lora.safetensors"
#   "/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/nail_new/final_lora.safetensors"
  # 在这里添加更多需要合并的 LoRA 文件路径...
  # "/path/to/your/lora5.safetensors"
)

# --- 输出 LoRA 文件路径 ---
# 在这里直接定义合并后的输出文件路径
output_file="/home/s414e2/CJH/Text-to-3D/LucidDreamer/final_merged_all_hardcoded.safetensors"


# --- 脚本逻辑 ---

# 获取输入文件数量
num_inputs=${#input_files[@]}

# 检查输入文件数量 (至少需要两个)
if [ "$num_inputs" -lt 2 ]; then
  echo "错误: 脚本内部定义的输入 LoRA 文件 ('input_files') 少于两个，无法合并。"
  echo "请编辑脚本并在 'input_files' 数组中添加至少两个文件路径。"
  exit 1
fi

# 检查输出文件路径是否已定义
if [ -z "$output_file" ]; then
    echo "错误: 脚本内部未定义输出 LoRA 文件路径 ('output_file')。"
    echo "请编辑脚本并设置 'output_file' 变量。"
    exit 1
fi

echo "--- 开始合并 LoRA 文件 ---"
echo "输入文件 (${num_inputs}) (来自脚本内部定义):"
for file in "${input_files[@]}"; do
  echo "  - $file"
  # 检查文件是否存在
  if [ ! -f "$file" ]; then
    echo "错误: 输入文件 '$file' 不存在。"
    exit 1
  fi
done
echo "输出文件 (来自脚本内部定义): $output_file"
echo "--------------------------"

# 获取输出目录
output_dir=$(dirname "$output_file")
# 确保输出目录存在
mkdir -p "$output_dir"
if [ $? -ne 0 ]; then
    echo "警告: 无法创建输出目录 '$output_dir'，如果目录不存在，合并可能会失败。"
fi

# 定义临时文件路径 (使用 .safetensors 后缀)
temp_file_current="${output_dir}/_temp_merge_current.safetensors"
temp_file_next="${output_dir}/_temp_merge_next.safetensors"

# 清理可能存在的旧临时文件
rm -f "$temp_file_current" "$temp_file_next"

# 1. 合并前两个文件
echo "[1/${num_inputs}] 正在合并: '${input_files[0]}' 和 '${input_files[1]}' -> '$temp_file_current'"
# 使用位置参数调用 lora_add
"$LORA_ADD_CMD" "${input_files[0]}" "${input_files[1]}" "$temp_file_current" --alpha_1 "$ALPHA_1" --alpha_2 "$ALPHA_2"
if [ $? -ne 0 ]; then
  echo "错误: 合并 '${input_files[0]}' 和 '${input_files[1]}' 时失败。"
  rm -f "$temp_file_current" # 清理失败的临时文件
  exit 1
fi

# 检查 lora_add 是否真的创建了输出文件
if [ ! -f "$temp_file_current" ]; then
  echo "严重错误: lora_add 命令似乎已成功执行 (退出码 0)，但预期的输出文件 '$temp_file_current' 未找到。"
  exit 1
fi


# 2. 循环合并剩余的文件
for (( i=2; i<num_inputs; i++ )); do
  current_input_file="${input_files[$i]}"

  echo "[$((i+1))/${num_inputs}] 正在合并: '$temp_file_current' 和 '$current_input_file' -> '$temp_file_next'"
  # 使用位置参数调用 lora_add
  "$LORA_ADD_CMD" "$temp_file_current" "$current_input_file" "$temp_file_next" --alpha_1 "$ALPHA_1" --alpha_2 "$ALPHA_2"
  if [ $? -ne 0 ]; then
    echo "错误: 合并 '$temp_file_current' 和 '$current_input_file' 时失败。"
    rm -f "$temp_file_current" "$temp_file_next" # 清理临时文件
    exit 1
  fi

  # 检查 lora_add 是否真的创建了输出文件
  if [ ! -f "$temp_file_next" ]; then
    echo "严重错误: lora_add 命令似乎已成功执行 (退出码 0)，但预期的输出文件 '$temp_file_next' 未找到。"
    rm -f "$temp_file_current" # 清理上一步的临时文件
    exit 1
  fi

  # 删除旧的当前临时文件 (现在是上一步的合并结果)
  rm -f "$temp_file_current"
  # 将新的合并结果 (next) 重命名为当前 (current)，为下一步做准备
  mv "$temp_file_next" "$temp_file_current"
  if [ $? -ne 0 ]; then
    echo "错误: 重命名临时文件 '$temp_file_next' 到 '$temp_file_current' 失败。"
    exit 1
  fi
done

# 3. 将最终的临时文件重命名为目标输出文件
echo "重命名最终结果 '$temp_file_current' -> '$output_file'"
# 使用 -f 选项强制覆盖，以防输出文件已存在
mv -f "$temp_file_current" "$output_file"
if [ $? -ne 0 ]; then
  echo "错误: 重命名最终输出文件失败。"
  # 尝试保留最后的临时文件以供检查
  echo "最后的临时文件 '$temp_file_current' 可能包含合并结果。"
  exit 1
fi

echo "--------------------------"
echo "成功合并所有 LoRA 文件到: $output_file"
echo "--- 合并完成 ---"

exit 0