import os
from safetensors import safe_open
from safetensors.torch import load_file

def view_safetensor_keys_method1(file_path):
    """
    方法1: 使用 safe_open 上下文管理器
    这种方法不会将整个文件加载到内存中，适合大文件
    """
    print(f"=== 方法1: 使用 safe_open ===")
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        print(f"文件中的键数量: {len(list(keys))}")
        print("所有键:")
        for key in f.keys():
            tensor_shape = f.get_tensor(key).shape
            print(f"  {key}: {tensor_shape}")

def view_safetensor_keys_method2(file_path):
    """
    方法2: 直接加载文件
    这种方法会将文件加载到内存中
    """
    print(f"\n=== 方法2: 直接加载文件 ===")
    tensors = load_file(file_path)
    print(f"文件中的键数量: {len(tensors)}")
    print("所有键:")
    for key, tensor in tensors.items():
        print(f"  {key}: {tensor.shape}")

def view_safetensor_metadata(file_path):
    """
    方法3: 查看元数据（如果有的话）
    """
    print(f"\n=== 方法3: 查看元数据 ===")
    with safe_open(file_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        if metadata:
            print("元数据:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("没有元数据")

def simple_list_keys(file_path):
    """
    简单方法: 只列出键名
    """
    print(f"\n=== 简单方法: 只列出键名 ===")
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"键列表: {keys}")
        return keys

# 使用示例
if __name__ == "__main__":
    # 替换为你的 safetensor 文件路径
    file_path = "/root/lora_train/multi_combine_20250617_211727/multi_teacher_distilled/final_multi_teacher_lora_step_5000.safetensors"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("请将 file_path 变量设置为你的 safetensor 文件路径")
        exit(1)
    
    try:
        # 使用不同方法查看键
        view_safetensor_keys_method1(file_path)
        view_safetensor_keys_method2(file_path)
        view_safetensor_metadata(file_path)
        
        # 获取键列表
        keys = simple_list_keys(file_path)
        
        # 可以进一步处理键，比如按名称排序
        print(f"\n=== 按字母顺序排序的键 ===")
        sorted_keys = sorted(keys)
        for key in sorted_keys:
            print(f"  {key}")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("请确保:")
        print("1. 文件路径正确")
        print("2. 已安装 safetensors 库: pip install safetensors")
        print("3. 文件是有效的 safetensor 格式")