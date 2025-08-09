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

# 定义自注意力机制
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, gcn_embed, clip_embed):
        # 确保所有张量的数据类型一致
        gcn_embed = gcn_embed.to(self.query_layer.weight.dtype)
        clip_embed = clip_embed.to(self.query_layer.weight.dtype)
        
        # 生成 Query, Key, Value
        query = self.query_layer(gcn_embed)  # (batch, embed_dim)
        key = self.key_layer(clip_embed)    # (batch, embed_dim)
        value = self.value_layer(clip_embed)

        # 注意力权重
        attn_weights = self.softmax(query * key)  # (batch, embed_dim)

        # 加权融合
        fused_embed = attn_weights * value + (1 - attn_weights) * gcn_embed
        return fused_embed