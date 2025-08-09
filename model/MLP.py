import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个函数将 dict 类型的嵌入转换为 tensor
def dict_to_tensor(embeddings_dict, device):
    keys = list(embeddings_dict.keys())
    embeddings_list = [embeddings_dict[key].to(device) for key in keys]
    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    return embeddings_tensor, keys

# 定义一个函数将 tensor 类型的嵌入转换回 dict
def tensor_to_dict(embeddings_tensor, keys):
    embeddings_dict = {}
    split_tensors = torch.split(embeddings_tensor, 1, dim=0)
    for key, tensor in zip(keys, split_tensors):
        embeddings_dict[key] = tensor  # 不使用 squeeze 保持维度不变
    return embeddings_dict

class EmbeddingFusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingFusionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, clip_embedding, gcn_embedding):
        # 将两个embedding拼接在一起
        combined = torch.cat((clip_embedding, gcn_embedding), dim=-1)
        # 通过MLP进行融合
        x = self.relu(self.fc1(combined))
        output = self.fc2(x)
        return output