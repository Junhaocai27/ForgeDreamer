import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义权重学习网络
class WeightLearningNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(WeightLearningNetwork, self).__init__()
        self.fc = nn.Linear(embedding_dim * 2, 2)
        self.softmax = nn.Softmax(dim=2)  # 修改为 dim=2

    def forward(self, clip_embedding, gcn_embedding):
        combined = torch.cat((clip_embedding, gcn_embedding), dim=2)  # 拼接在最后一个维度
        weights = self.fc(combined)
        weights = self.softmax(weights)
        return weights

# 定义融合模型
class EmbeddingFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingFusion, self).__init__()
        self.weight_network = WeightLearningNetwork(embedding_dim)

    def forward(self, clip_embedding, gcn_embedding):
        weights = self.weight_network(clip_embedding, gcn_embedding)
        clip_weight = weights[:, :, 0].unsqueeze(2)  # 修改为 unsqueeze(2)
        gcn_weight = weights[:, :, 1].unsqueeze(2)  # 修改为 unsqueeze(2)
        fused_embedding = (clip_weight * 2) * clip_embedding + gcn_weight * gcn_embedding
        return fused_embedding