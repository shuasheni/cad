from joint_predict.datasets.joint_graph_dataset import JointGraphDataset
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torchvision.ops.focal_loss import sigmoid_focal_loss

from joint_predict.utils import metrics

from torch_geometric.nn import GCNConv



def mlp(inp_dim, hidden_dim, out_dim, num_layers=1, batch_norm=False):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Linear(inp_dim if i == 0 else hidden_dim, hidden_dim, bias=not batch_norm))
        if batch_norm:
            modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.ELU())
    modules.append(nn.Linear(hidden_dim, out_dim, bias=True))
    return nn.Sequential(*modules)


def cnn2d(inp_channels, hidden_channels, out_dim, num_layers=1, batch_norm=False):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv2d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            modules.append(nn.BatchNorm2d(hidden_channels))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


def cnn1d(inp_channels, hidden_channels, out_dim, num_layers=1, batch_norm=False):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv1d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=not batch_norm))
        if batch_norm:
            modules.append(nn.BatchNorm1d(hidden_channels))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool1d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


class EdgeDotProductMPN(nn.Module):
    """
    Performs message passing in the joint graph (connecting the two bodies)
    to web_app edge logits by a dot product
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src, :]
        x_dst = x[dst, :]
        return (x_src * x_dst).sum(-1)


class InnerProductDecoderWithNegativeSampling(nn.Module):
    """
    内积解码器 + 负采样，用于高效计算正负样本的边logits。
    """

    def __init__(self, num_neg_samples=5):
        super().__init__()
        self.num_neg_samples = num_neg_samples  # 每个正样本对应的负样本数

    def forward(self, x, edge_index):
        """
        输入:
        - x: 合并后的节点特征 [N_total, D]
        - edge_index: 正样本边索引 [2, E_pos]
        输出:
        - logits: 正负样本边的logits [E_pos + E_neg]
        - labels: 正负样本标签 [E_pos + E_neg] (1: 正样本, 0: 负样本)
        """
        # ---------------------- 正样本计算 ----------------------
        src_pos, dst_pos = edge_index[0], edge_index[1]
        x_src_pos = x[src_pos, :]  # [E_pos, D]
        x_dst_pos = x[dst_pos, :]  # [E_pos, D]
        pos_logits = (x_src_pos * x_dst_pos).sum(-1)  # 正样本logits [E_pos]

        # ---------------------- 负采样 ----------------------
        num_pos = src_pos.size(0)
        num_neg = num_pos * self.num_neg_samples  # 负样本总数

        # 随机生成负样本的源节点和目标节点索引（避免与正样本重复）
        num_nodes = x.size(0)
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=x.device)  # [E_neg]
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=x.device)  # [E_neg]

        # 计算负样本logits
        x_neg_src = x[neg_src, :]  # [E_neg, D]
        x_neg_dst = x[neg_dst, :]  # [E_neg, D]
        neg_logits = (x_neg_src * x_neg_dst).sum(-1)  # [E_neg]

        # ---------------------- 合并结果 ----------------------
        logits = torch.cat([pos_logits, neg_logits], dim=0)  # [E_pos + E_neg]
        labels = torch.cat([
            torch.ones_like(pos_logits),  # 正样本标签为1
            torch.zeros_like(neg_logits)  # 负样本标签为0
        ], dim=0)

        return logits, labels  # 返回logits和标签

class GraphMatchingMPN(nn.Module):
    """
    图匹配网络（Graph Matching Network）用于链接预测，通过跨图注意力增强节点特征。
    """

    def __init__(self, hidden_dim, heads=4, batch_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        # 跨图多头注意力层
        self.attn = nn.MultiheadAttention(hidden_dim, heads)

        # 特征融合MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 可选的相似度计算方式（双线性变换）
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        """
        输入:
        - x: 合并后的节点特征 [N_total, D]
        - edge_index: 边索引 [2, E]，其中 src 属于图1，dst 属于图2
        输出:
        - logits: 边存在概率 [E]
        """
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src, :]  # 图1的节点特征 [E, D]
        x_dst = x[dst, :]  # 图2的节点特征 [E, D]

        # 跨图注意力（Query=src, Key/Value=dst）
        x_src_seq = x_src.unsqueeze(0)  # [1, E, D]
        x_dst_seq = x_dst.unsqueeze(0)  # [1, E, D]
        attn_out, _ = self.attn(x_src_seq, x_dst_seq, x_dst_seq)  # [1, E, D]
        attn_out = attn_out.squeeze(0)  # [E, D]

        # 特征融合：拼接原始特征与注意力输出
        combined = torch.cat([x_src, attn_out], dim=-1)  # [E, 2D]
        enhanced_src = self.mlp(combined)  # [E, D]

        # 双线性变换计算相似度
        logits = (enhanced_src @ self.W @ x_dst.T).diag()  # [E]
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEdgePredictor(nn.Module):
    """
    基于图卷积网络（GCN）的边预测模型，输入输出格式与 EdgeMLPMPN/EdgeDotProductMPN 兼容。
    包含以下步骤：
    1. 多层 GCN 进行节点嵌入
    2. 可选的点积或 MLP 计算边 logits
    """

    def __init__(self, in_channels, hidden_channels, out_channels=1,
                 num_layers=2, method="dot_product", batch_norm=False):
        super().__init__()
        self.method = method
        self.num_layers = num_layers

        # ---------------------- GCN 消息传递层 ----------------------
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # ---------------------- 边 logits 计算层 ----------------------
        if method == "dot_product":
            # 直接使用点积（无需参数）
            self.edge_decoder = lambda x_src, x_dst: (x_src * x_dst).sum(dim=-1)
        elif method == "mlp":
            # 使用 MLP 计算 logits
            self.edge_decoder = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

    def forward(self, x, edge_index):
        """
        输入:
        - x: 节点特征矩阵 [num_nodes, in_channels]
        - edge_index: 边索引 [2, num_edges]
        输出:
        - logits: 边存在概率 [num_edges]
        """
        # ---------------------- GCN 消息传递 ----------------------
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)

        # ---------------------- 边 logits 计算 ----------------------
        src, dst = edge_index[0], edge_index[1]
        x_src = h[src]  # [num_edges, hidden_channels]
        x_dst = h[dst]  # [num_edges, hidden_channels]

        if self.method == "dot_product":
            logits = (x_src * x_dst).sum(dim=-1)  # [num_edges]
        elif self.method == "mlp":
            x_pair = torch.cat([x_src, x_dst], dim=-1)  # [num_edges, 2*hidden_channels]
            logits = self.edge_decoder(x_pair).squeeze(-1)  # [num_edges]

        return logits

class EdgeMLPMPN(nn.Module):
    """
    Performs message passing in the joint graph (connecting the two bodies)
    to web_app edge logits by using an MLP (1x1 convolutions)
    """
    def __init__(self, in_channels, hidden_channels, batch_norm=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(2 * in_channels, hidden_channels, kernel_size=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ELU())
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ELU())
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        src, tgt = edge_index[0], edge_index[1]
        x_src = x[src, :]
        x_dst = x[tgt, :]
        x = torch.cat([x_src, x_dst], dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.net(x)
        return x.squeeze()


def prepare_features_for_joint_graph(x1, x2, jg):
    joint_graph_unbatched = jg.to_data_list()
    num_nodes_graph1 = [item.num_nodes_graph1 for item in joint_graph_unbatched]
    num_nodes_graph2 = [item.num_nodes_graph2 for item in joint_graph_unbatched]

    start1 = 0
    start2 = 0
    concat_x = []
    for i in range(len(joint_graph_unbatched)):
        size1_i = num_nodes_graph1[i]
        size2_i = num_nodes_graph2[i]
        end1 = start1 + size1_i
        end2 = start2 + size2_i
        x1_i = x1[start1:end1]
        x2_i = x2[start2:end2]
        concat_x.append(x1_i)
        concat_x.append(x2_i)
        start1 = end1
        start2 = end2
    return torch.cat(concat_x, dim=0)


class PostJointNet(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, reduction="sum", method="mlp", batch_norm=False):
        super(PostJointNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.reduction = reduction
        if self.reduction is None:
            self.reduction = "sum"
        self.method = method
        if method not in ("mm", "mlp", "gm", "gcn"):
            raise NotImplemented("Expected 'method' to be 'mm' or 'mlp' or 'gm'")
        self.dropout = nn.Dropout(dropout)

        if self.method == "mm":
            self.mpn = EdgeDotProductMPN()
        elif self.method == "mlp":
            self.mpn = EdgeMLPMPN(hidden_dim, hidden_dim, batch_norm=batch_norm)
        elif method == "gm":
            self.mpn = GraphMatchingMPN(hidden_dim, heads=4, batch_norm=batch_norm)
        elif method == "ipn":
            self.mpn = InnerProductDecoderWithNegativeSampling(num_neg_samples=5)
        elif method == "gcn":
            self.mpn = GCNEdgePredictor(in_channels=64, hidden_channels=256, num_layers=3, method="mlp", batch_norm=True)


    def forward(self, x1, x2, jg):
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x = prepare_features_for_joint_graph(x1, x2, jg)
        logits = self.mpn(x, jg.edge_index)
        return logits


def _get_edge_node_indices(g):
    """Get the indices of graph nodes corresponding to B-rep edges"""
    edge_indices = torch.where(g["is_face"] <= 0.5)[0].long()
    return edge_indices


def _get_face_node_indices(g):
    """Get the indices of graph nodes corresponding to B-rep faces"""
    face_indices = torch.where(g["is_face"] > 0.5)[0].long()
    return face_indices


class PreJointNetFace(nn.Module):
    def __init__(
        self,
        hidden_dim,
        input_features,
        batch_norm=False,
        method="mlp",
    ):
        super(PreJointNetFace, self).__init__()
        assert method in ("mlp", "cnn")
        self.method = method
        half_hidden_dim = int(hidden_dim / 2)
        # Turn the comma separated string e.g. "points,entity_types,is_face,length"
        # into lists for each feature type, i.e.
        # - Features used in the UV grid, i.e. points, normals, trimming_mask
        # - Features related to each B-Rep entity e.g. area, length
        feat_lists = JointGraphDataset.parse_input_features_arg(input_features, input_feature_type="face")
        self.input_features, self.grid_input_features, self.entity_input_features = feat_lists
        # Calculate the total size of each feature list
        self.grid_feat_size = JointGraphDataset.get_input_feature_size(self.grid_input_features, input_feature_type="face")
        self.ent_feat_size = JointGraphDataset.get_input_feature_size(self.entity_input_features, input_feature_type="face")

        # Setup the layers
        if self.grid_feat_size > 0:
            grid_dim = half_hidden_dim
            # If we don't have other features use the whole hidden dimension size
            if self.ent_feat_size == 0:
                grid_dim = hidden_dim
            self.grid_dim = grid_dim
            if method == "mlp":
                self.model_pre_grid = mlp(self.grid_feat_size, grid_dim, grid_dim, num_layers=2, batch_norm=batch_norm)
            else:
                channels = self.grid_feat_size // JointGraphDataset.grid_len
                self.model_pre_grid = cnn2d(channels, grid_dim, grid_dim, num_layers=3, batch_norm=batch_norm)
        if self.ent_feat_size > 0:
            ent_dim = half_hidden_dim
            # If we don't have other features use the whole hidden dimension size
            if self.grid_feat_size == 0:
                ent_dim = hidden_dim
            self.ent_dim = ent_dim
            self.model_pre_ent = mlp(self.ent_feat_size, ent_dim, ent_dim, num_layers=2, batch_norm=batch_norm)

    def get_entity_features(self, g):
        """Get the entity features that were requested"""
        ent_list = []
        for input_feature in self.entity_input_features:
            feat = g[input_feature].float()
            # Make all features 2D
            if len(feat.shape) == 1:
                feat = feat.unsqueeze(1)
            if input_feature == "entity_types":
                feat = feat[:, :8]  # Keep only the face entity types
            ent_list.append(feat)
        return torch.cat(ent_list, dim=1).float()

    def get_grid_features(self, g):
        """Get the grid features that were requested"""
        grid_list = []
        if "points" in self.grid_input_features:
            grid_list.append(g.x[:, :, :, :3])
        if "normals" in self.grid_input_features:
            grid_list.append(g.x[:, :, :, 3:6])
        if "trimming_mask" in self.grid_input_features:
            grid_list.append(g.x[:, :, :, 6:])

        grid = torch.cat(grid_list, dim=-1)
        if self.method == "mlp":
            # If we have an MLP, then flatten the grid features
            grid = grid.view(g.num_nodes, -1)
        elif self.method == "cnn":
            # If we have an CNN, then bring the channels into the 2nd dimension
            grid = grid.permute(0, 3, 1, 2)
        return grid

    def forward_one_graph(self, g):
        face_node_indices = _get_face_node_indices(g)
        device = g.edge_index.device
        # We have both grid and entity features
        if self.ent_feat_size > 0 and self.grid_feat_size > 0:
            x = torch.zeros(g.num_nodes, self.grid_dim + self.ent_dim, dtype=torch.float, device=device)
            ent = self.get_entity_features(g)
            grid = self.get_grid_features(g)
            ent_faces = ent[face_node_indices, :]
            grid_faces = grid[face_node_indices, :]
            x_ent = self.model_pre_ent(ent_faces)
            x_grid = self.model_pre_grid(grid_faces)
            x[face_node_indices, :] = torch.cat((x_grid, x_ent), dim=1)
        # We have no grid features, so just use the entity features
        elif self.ent_feat_size > 0:
            ent = self.get_entity_features(g)
            ent_faces = ent[face_node_indices, :]
            x = torch.zeros(g.num_nodes, self.ent_dim, dtype=torch.float, device=device)
            x[face_node_indices, :] = self.model_pre_ent(ent_faces)
        # We have no entity features, so just use the grid
        elif self.grid_feat_size > 0:
            grid = self.get_grid_features(g)
            grid_faces = grid[face_node_indices, :]
            x = torch.zeros(g.num_nodes, self.grid_dim, dtype=torch.float, device=device)
            x[face_node_indices, :] = self.model_pre_grid(grid_faces)
        return x

    def forward(self, g1, g2):
        x1 = self.forward_one_graph(g1)
        x2 = self.forward_one_graph(g2)
        return x1, x2


class PreJointNetEdge(nn.Module):
    def __init__(
        self,
        hidden_dim,
        input_features,
        batch_norm=False,
        method="mlp",
    ):
        super(PreJointNetEdge, self).__init__()
        assert method in ("mlp", "cnn")
        self.method = method
        half_hidden_dim = int(hidden_dim / 2)
        # Turn the comma separated string e.g. "points,entity_types,is_face,length"
        # into lists for each feature type, i.e.
        # - Features used in the UV grid, i.e. points, normals, trimming_mask
        # - Features related to each B-Rep entity e.g. area, length
        feat_lists = JointGraphDataset.parse_input_features_arg(input_features, input_feature_type="edge")
        self.input_features, self.grid_input_features, self.entity_input_features = feat_lists
        # Calculate the total size of each feature list
        self.grid_feat_size = JointGraphDataset.get_input_feature_size(self.grid_input_features, input_feature_type="edge")
        self.ent_feat_size = JointGraphDataset.get_input_feature_size(self.entity_input_features, input_feature_type="edge")

        # layers
        if self.grid_feat_size > 0:
            grid_dim = half_hidden_dim
            # If we don't have other features use the whole hidden dimension size
            if self.ent_feat_size == 0:
                grid_dim = hidden_dim
            self.grid_dim = grid_dim
            if self.method == "mlp":
                self.model_pre_grid = mlp(self.grid_feat_size, grid_dim, grid_dim, num_layers=2, batch_norm=batch_norm)
            else:
                channels = self.grid_feat_size // JointGraphDataset.grid_size
                self.model_pre_grid = cnn1d(channels, grid_dim, grid_dim, num_layers=3, batch_norm=batch_norm)
        if self.ent_feat_size > 0:
            ent_dim = half_hidden_dim
            # If we don't have other features use the whole hidden dimension size
            if self.grid_feat_size == 0:
                ent_dim = hidden_dim
            self.ent_dim = ent_dim
            self.model_pre_ent = mlp(self.ent_feat_size, ent_dim, ent_dim, num_layers=2, batch_norm=batch_norm)

    def get_entity_features(self, g):
        """Get the entity features that were requested"""
        ent_list = []
        for input_feature in self.entity_input_features:
            feat = g[input_feature].float()
            # Make all features 2D
            if len(feat.shape) == 1:
                feat = feat.unsqueeze(1)
            if input_feature == "entity_types":
                feat = feat[:, 8:]  # Keep only the edge entity types
            ent_list.append(feat)

        return torch.cat(ent_list, dim=1).float()

    def get_grid_features(self, g):
        """Get the grid features that were requested"""
        # Edge grid features are repeated along the rows,
        # so we only use the feature in the first row
        grid_list = []
        if "points" in self.grid_input_features:
            grid_list.append(g.x[:, 0, :, :3])
        if "tangents" in self.grid_input_features:
            grid_list.append(g.x[:, 0, :, 3:6])
        grid = torch.cat(grid_list, dim=-1)
        if self.method == "mlp":
            # If we have an MLP, then flatten the grid features
            grid = grid.view(g.num_nodes, -1)
        elif self.method == "cnn":
            # If we have an CNN, then bring the channels into the 2nd dimension
            grid = grid.permute(0, 2, 1)
        return grid

    def forward_one_graph(self, g):
        edge_node_indices = _get_edge_node_indices(g)
        device = g.edge_index.device
        # We have both grid and entity features
        if self.ent_feat_size > 0 and self.grid_feat_size > 0:
            x = torch.zeros(g.num_nodes, self.grid_dim + self.ent_dim, dtype=torch.float, device=device)
            ent = self.get_entity_features(g)
            grid = self.get_grid_features(g)
            ent_edges = ent[edge_node_indices, :]
            grid_edges = grid[edge_node_indices, :]
            x_ent = self.model_pre_ent(ent_edges)
            x_grid = self.model_pre_grid(grid_edges)
            x[edge_node_indices, :] = torch.cat((x_grid, x_ent), dim=1)
        # We have no grid features, so just use the entity features
        elif self.ent_feat_size > 0:
            ent = self.get_entity_features(g)
            ent_edges = ent[edge_node_indices, :]
            x = torch.zeros(g.num_nodes, self.ent_dim, dtype=torch.float, device=device)
            x[edge_node_indices, :] = self.model_pre_ent(ent_edges)
        # We have no entity features, so just use the grid
        elif self.grid_feat_size > 0:
            grid = self.get_grid_features(g)
            grid_edges = grid[edge_node_indices, :]
            x = torch.zeros(g.num_nodes, self.grid_dim, dtype=torch.float, device=device)
            x[edge_node_indices, :] = self.model_pre_grid(grid_edges)
        return x

    def forward(self, g1, g2):
        x1 = self.forward_one_graph(g1)
        x2 = self.forward_one_graph(g2)
        return x1, x2


class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, dropout, mpn, batch_norm=False):
        super(GAT, self).__init__()
        if mpn == "gat":
            self.conv1 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
        elif mpn == "gatv2":
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
        elif mpn == "gcn":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            raise Exception("Unknown mpn argument")
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edges_idx):
        x = self.dropout(x)
        x = self.conv1(x, edges_idx)
        if self.batch_norm:
            x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edges_idx)
        return x

class GCN_MPN(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class JoinABLe(nn.Module):
    def __init__(
        self,
        hidden_dim,
        input_features,
        dropout=0.0,
        mpn="gatv2",
        batch_norm=False,
        reduction="sum",
        post_net="mlp",
        pre_net="mlp",
    ):
        super(JoinABLe, self).__init__()
        self.reduction = reduction
        self.pre_face = PreJointNetFace(hidden_dim, input_features, batch_norm=batch_norm, method=pre_net)
        self.pre_edge = PreJointNetEdge(hidden_dim, input_features, batch_norm=batch_norm, method=pre_net)
        # self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.mpn = GAT(hidden_dim, dropout, mpn, batch_norm=batch_norm)
        self.post = PostJointNet(hidden_dim, dropout=dropout, reduction=reduction, method=post_net, batch_norm=batch_norm)

    def forward(self, g1, g2, jg):
        # Compute the features for the is_face nodes, and set the rest to zero
        x1_face, x2_face = self.pre_face(g1, g2)
        # Compute the features for the NOT is_face nodes, and set the rest to zero
        x1_edge, x2_edge = self.pre_edge(g1, g2)

        # print(f"x1_face shape: {x1_face.shape}")
        # print(f"x1_edge shape: {x1_edge.shape}")
        # print(f"x2_face shape: {x2_face.shape}")
        # print(f"x2_edge shape: {x2_edge.shape}")
        # Sum the two features to populate the edge and face features at the right locations
        x1 = x1_edge + x1_face
        x2 = x2_edge + x2_face

        # print(f"x1 shape: {x1.shape}")
        #
        # print(f"x2 shape: {x2.shape}")
        # Projection to common space
        # x1 = self.proj(x1)
        # x2 = self.proj(x2)
        # Message passing
        x1 = self.mpn(x1, g1.edge_index)
        x2 = self.mpn(x2, g2.edge_index)


        # print(f"h1 shape: {x1.shape}")
        # print(f"h2 shape: {x2.shape}")
        # Pass to post-net
        x = self.post(x1, x2, jg)

        # print(f"final shape: {x.shape}")
        return x

    def soft_cross_entropy(self, input, target):
        logprobs = F.log_softmax(input, dim=-1)
        # loss = -(target * logprobs).sum()
        loss = -torch.dot(target, logprobs)
        return loss

    def compute_loss(self, args, x, joint_graph):
        joint_graph_unbatched = joint_graph.to_data_list()
        batch_size = len(joint_graph_unbatched)
        size_of_each_joint_graph = [np.product(list(item.edge_attr.shape)) for item in joint_graph_unbatched]
        num_nodes_graph1 = [item.num_nodes_graph1 for item in joint_graph_unbatched]
        num_nodes_graph2 = [item.num_nodes_graph2 for item in joint_graph_unbatched]

        # Compute loss individually with each joint in the batch
        start = 0
        loss_clf = 0
        loss_sym = 0
        for i in range(len(joint_graph_unbatched)):
            size_i = size_of_each_joint_graph[i]
            end = start + size_i
            x_i = x[start:end]
            labels_i = joint_graph_unbatched[i].edge_attr
            # Classification loss
            if args.loss == "bce":
                loss_clf += self.bce_loss(x_i, labels_i, pos_weight=args.pos_weight)
            elif args.loss == "mle":
                loss_clf += self.mle_loss(x_i, labels_i)
            # Symmetric loss
            loss_sym += self.symmetric_loss(x_i, labels_i, num_nodes_graph1[i], num_nodes_graph2[i])
            start = end

        # Total loss
        loss = loss_clf + loss_sym
        if self.reduction == "mean":
            loss = loss / float(batch_size)
        else:
            # Do nothing: Loss is already reduced by sum
            pass
        return loss

    def mle_loss(self, x, labels):
        # Normalize the ground truth matrix into a PDF
        labels = labels / labels.sum()
        labels = labels.view(-1)
        # Compare the predicted and ground truth PDFs
        loss = self.soft_cross_entropy(x, labels)
        return loss

    def focal_loss(self, x, ys_matrix, gamma=2.0, alpha=0.25):
        x = x.unsqueeze(1)
        ys_matrix = ys_matrix.flatten().unsqueeze(1).float()
        return sigmoid_focal_loss(x, ys_matrix, reduction="sum", gamma=gamma, alpha=alpha)

    def bce_loss(self, x, ys_matrix, pos_weight=200.0):
        x = x.unsqueeze(1)
        ys_matrix = ys_matrix.flatten().float().unsqueeze(1)
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([pos_weight]).to(x.device))
        return loss_fn(x, ys_matrix)

    def symmetric_loss(self, x, labels, n1, n2):
        x_2d = x.view(n1, n2)
        ys_2d = torch.nonzero(labels.view(n1, n2)).to(x.device)
        loss1 = F.cross_entropy(x_2d[ys_2d[:, 0], :], ys_2d[:, 1], reduction="mean")
        loss2 = F.cross_entropy(x_2d[:, ys_2d[:, 1]].transpose(0, 1), ys_2d[:, 0], reduction="mean")
        loss = 0.5 * (loss1 + loss2)
        return loss

    def accuracy(self, prob, labels, t):
        gt = labels.view(-1)
        pred = (prob >= t).int()
        true_positive = torch.sum(torch.logical_and(gt == 1, pred == 1)).item()
        false_negative = torch.sum(torch.logical_and(gt == 1, pred == 0)).item()
        false_positive = torch.sum(torch.logical_and(gt == 0, pred == 1)).item()
        true_negative = torch.sum(torch.logical_and(gt == 0, pred == 0)).item()
        return [true_positive, false_negative, false_positive, true_negative]

    def hit_at_top_k(self, prob, ys, k):
        pred_idx = torch.argsort(-prob)
        hit = 0
        for i in range(torch.min([k, prob.shape[0]])):
            if pred_idx[i] in ys:
                hit += 1
        num_joints = ys.shape[0]
        hit_best = torch.min([num_joints, k])
        return [hit, hit_best, num_joints]

    def precision_at_top_k(self, logits, labels, n1, n2, k=None):
        logits = logits.view(n1, n2)
        labels = labels.view(n1, n2)
        if k is None:
            k = metrics.get_k_sequence()
        return metrics.hit_at_top_k(logits, labels, k=k)
