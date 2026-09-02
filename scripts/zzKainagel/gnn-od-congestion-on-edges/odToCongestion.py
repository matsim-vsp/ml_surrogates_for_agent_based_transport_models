import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ------------------------------------------------------------
# 1. TOY GRAPH (Straßennetz)
# ------------------------------------------------------------

# 4 Knoten, 5 gerichtete Kanten
edge_index = torch.tensor([
    [0, 1, 2, 3, 1],
    [1, 2, 3, 0, 3]
], dtype=torch.long)

num_nodes = 4
num_edges = edge_index.shape[1]

# ------------------------------------------------------------
# 2. OD-MATRIX → flatten als Input
# ------------------------------------------------------------

# 4x4 OD matrix (toy example)
od_matrix = torch.tensor([
    [0, 5, 2, 1],
    [3, 0, 0, 2],
    [1, 1, 0, 4],
    [2, 0, 3, 0]
], dtype=torch.float)

od_vector = od_matrix.flatten()  # shape [16]

# ------------------------------------------------------------
# 3. Node features
# ------------------------------------------------------------

# simple encoding: every node gets OD row + OD col info
node_features = torch.zeros((num_nodes, 2))

node_features[:, 0] = od_matrix.sum(dim=1)  # outgoing demand
node_features[:, 1] = od_matrix.sum(dim=0)  # incoming demand

# ------------------------------------------------------------
# 4. Synthetic target: edge congestion
# ------------------------------------------------------------

# pretend "true congestion" depends on connected nodes
true_congestion = torch.tensor([
    10.0, 12.0, 15.0, 8.0, 11.0
]).view(-1, 1)

# ------------------------------------------------------------
# 5. PyG Data object
# ------------------------------------------------------------

data = Data(
    x=node_features,
    edge_index=edge_index,
    y=true_congestion,
    od=od_vector
)

# ------------------------------------------------------------
# 6. GNN Model
# ------------------------------------------------------------

class ODGraphModel(nn.Module):

    def __init__(self, od_dim):
        super().__init__()

        # embed OD vector into nodes (global → local injection)
        self.od_encoder = nn.Linear(od_dim, 8)

        # graph convolution layers
        self.conv1 = GCNConv(2 + 8, 16)
        self.conv2 = GCNConv(16, 16)

        # edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        # --------------------------------------------------------
        # inject OD information into each node
        # --------------------------------------------------------
        od_emb = self.od_encoder(data.od)
        x = torch.cat([x, od_emb.repeat(x.size(0), 1)], dim=1)

        # --------------------------------------------------------
        # GNN message passing
        # --------------------------------------------------------
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # --------------------------------------------------------
        # edge-level prediction
        # --------------------------------------------------------
        src, dst = edge_index

        edge_feat = torch.cat([x[src], x[dst]], dim=1)

        out = self.edge_mlp(edge_feat)

        return out

# ------------------------------------------------------------
# 7. Training setup
# ------------------------------------------------------------

model = ODGraphModel(od_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# ------------------------------------------------------------
# 8. Training loop
# ------------------------------------------------------------

for epoch in range(200):

    pred = model(data)

    loss = loss_fn(pred, data.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.4f}")