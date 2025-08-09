import os
import re

def find_matching_lora(input_text, view_type, lora_base_dir="/home/s414e2/CJH/Text-to-3D/LucidDreamer/custom_example/new_experiment/screw_new_ex"):
    """
    从输入文本中检测<>内的触发词并根据视角返回匹配的LoRA文件路径
    
    参数:
        input_text (str): 用户输入的文本提示
        view_type (str): 相机视角，如 'front' 或 'up'
        lora_base_dir (str): LoRA文件的基础目录路径
    
    返回:
        str 或 "": 匹配的LoRA文件的完整路径，如果没有匹配则返回空字符串
    """
    # 查找<>内的触发词
    trigger_pattern = re.compile(r'<([^>]+)>')
    matches = trigger_pattern.findall(input_text.lower())
    
    if not matches:
        print(f"未在文本中找到<>包围的触发词: '{input_text}'")
        return ""
    
    # 提取触发词（用于日志显示）
    trigger_word = matches[0].strip()
    base_trigger = trigger_word.split('_')[0] if '_' in trigger_word else trigger_word
    
    # 尝试直接访问视角目录
    view_dir = os.path.join(lora_base_dir, view_type)
    
    if os.path.exists(view_dir):
        # 首先检查视角目录下是否有final_lora.safetensors文件
        lora_file_path = os.path.join(view_dir, "final_lora.safetensors")
        if os.path.exists(lora_file_path):
            print(f"找到触发词 '<{trigger_word}>' 对应的 {view_type} 视角LoRA文件: final_lora.safetensors")
            return lora_file_path
        
        # 如果没有，检查视角目录下的所有.safetensors文件
        try:
            lora_files = [f for f in os.listdir(view_dir) if f.endswith('.safetensors')]
            if lora_files:
                # 尝试查找与触发词相关的文件
                for lora_file in lora_files:
                    file_name = os.path.splitext(lora_file)[0].lower()
                    if base_trigger in file_name:
                        print(f"找到触发词 '<{trigger_word}>' 对应的 {view_type} 视角LoRA文件: {lora_file}")
                        return os.path.join(view_dir, lora_file)
                
                # 如果没有找到匹配的文件，返回第一个.safetensors文件
                lora_file_path = os.path.join(view_dir, lora_files[0])
                print(f"未找到精确匹配 '{base_trigger}' 的文件，使用视角目录中的第一个文件: {lora_files[0]}")
                return lora_file_path
        except Exception as e:
            print(f"读取目录出错 {view_dir}: {str(e)}")
    
    # 尝试备用目录结构: /base_dir/view_type/final_lora
    final_lora_dir = os.path.join(view_dir, "final_lora")
    if os.path.exists(final_lora_dir):
        try:
            lora_files = [f for f in os.listdir(final_lora_dir) if f.endswith('.safetensors')]
            if lora_files:
                # 尝试查找与触发词相关的文件
                for lora_file in lora_files:
                    file_name = os.path.splitext(lora_file)[0].lower()
                    if base_trigger in file_name:
                        print(f"找到触发词 '<{trigger_word}>' 对应的 {view_type} 视角LoRA文件: {lora_file}")
                        return os.path.join(final_lora_dir, lora_file)
                
                # 如果没有找到匹配的文件，返回第一个.safetensors文件
                lora_file_path = os.path.join(final_lora_dir, lora_files[0])
                print(f"未找到精确匹配 '{base_trigger}' 的文件，使用final_lora目录中的第一个文件: {lora_files[0]}")
                return lora_file_path
        except Exception as e:
            print(f"读取目录出错 {final_lora_dir}: {str(e)}")
    
    # 如果以上方法都失败，尝试旧的目录结构
    old_structure_dir = os.path.join(lora_base_dir, base_trigger, view_type)
    if os.path.exists(old_structure_dir):
        lora_file_path = os.path.join(old_structure_dir, "final_lora.safetensors")
        if os.path.exists(lora_file_path):
            print(f"使用旧目录结构找到的LoRA文件: {lora_file_path}")
            return lora_file_path
        
        try:
            lora_files = [f for f in os.listdir(old_structure_dir) if f.endswith('.safetensors')]
            if lora_files:
                lora_file_path = os.path.join(old_structure_dir, lora_files[0])
                print(f"使用旧目录结构找到的LoRA文件: {lora_file_path}")
                return lora_file_path
        except Exception as e:
            print(f"读取旧目录结构出错: {str(e)}")
    
    # 如果没有找到任何匹配项
    print(f"在任何已知的目录结构中都未能找到视角 '{view_type}' 的LoRA文件")
    return ""

def select_lora_for_text(text, view_type="front"):
    """
    为给定文本和视角选择合适的LoRA文件路径
    
    参数:
        text (str): 输入的文本提示
        view_type (str, optional): 相机视角，如 'front' 或 'up'。默认为'front'
    
    返回:
        str 或 空字符串: 匹配的LoRA文件的完整路径，如果没有匹配则返回空字符串
    """
    lora_path = find_matching_lora(text, view_type)
    
    if lora_path:
        return lora_path
    else:
        print(f"警告: 没有找到与文本 '{text}' 和视角 '{view_type}' 匹配的LoRA文件")
        return ""