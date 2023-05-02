import torch
import torch.nn as nn
from torch import Tensor


class PredicateLevelEncoder(nn.Module):
    def __init__(self, feature3d_dim, hidden_dim, semantics_dim, useless_objects):
        super(PredicateLevelEncoder, self).__init__()
        self.linear_layer = nn.Linear(feature3d_dim, hidden_dim)
        # ZFC w、W、U、b为可学习参数
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.b = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.w = nn.Linear(hidden_dim, 1)
        self.inf = 1e5
        self.useless_objects = useless_objects

        self.bilstm = nn.LSTM(input_size=hidden_dim + hidden_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)

    def forward(self, features3d: Tensor, objects: Tensor, objects_mask: Tensor):
        """

        Args:
            features3d: (bsz, sample_numb, 3d_dim)
            objects: (bsz, max_objects, hidden_dim)
            objects_mask: (bsz, max_objects_per_video)

        Returns:
            action_features: (bsz, sample_numb, hidden_dim * 2)
            action_pending: (bsz, semantics_dim)
        """
        # ZFC features3d 即 motion feature --> m_i；# sample_numb = 15 个 clips
        sample_numb = features3d.shape[1]
        # ZFC 建立 feature3d_dim --> hidden_dim 的投影关系
        features3d = self.linear_layer(features3d)  # (bsz, sample_numb, hidden_dim)
        # ZFC 计算 W^T_a * m_i
        Wf3d = self.W(features3d)  # (bsz, sample_numb, hidden_dim)
        # ZFC 计算 U^T_a * e_k
        Uobjs = self.U(objects)  # (bsz, max_objects, hidden_dim)
        # ZFC 计算 W^T_a * m_i + U^T_a * e_k + b_a
        attn_feat = Wf3d.unsqueeze(2) + Uobjs.unsqueeze(1) + self.b  # (bsz, sample_numb, max_objects, hidden_dim)
        # 计算 w^T_a * tanh --> \hat α_{i,k}
        attn_weights = self.w(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        objects_mask = objects_mask[:, None, :, None].repeat(1, sample_numb, 1, 1)  # (bsz, sample_numb, max_objects_per_video, 1)
        if self.useless_objects:
            attn_weights = attn_weights - objects_mask.float() * self.inf
        # ZFC softmax --> α_{i,k}
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        # ZFC sun(α_{i,k} * e_k) --> m^e_i
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
        # ZFC m_i、m^e_i 拼接
        features = torch.cat([features3d, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 2)
        # ZFC 利用 BiLSTM 编码
        output, states = self.bilstm(features)  # (bsz, sample_numb, hidden_dim)
        # ZFC MAX_POOLING
        action = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        # ZFC 得到 predicate language head
        action_pending = self.fc_layer(action)  # (bsz, semantics_dim)
        # ZFC 得到 action feature
        action_features = output  # (bsz, sample_numb, hidden_dim)

        return action_features, action_pending



