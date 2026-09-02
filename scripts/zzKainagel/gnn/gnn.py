import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --------------------------------------------------
# Graph definieren
# --------------------------------------------------

# Knotenfeatures:
# 3 Knoten, jeweils 2 Features

x = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 1.0]
])

# Kanten:
#
# 0 <-> 1
# 1 <-> 2

edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
])
# (this encodes from top to bottom!  I.e. 0->1, 1->0, etc. etc.  There is a "contiguous" method in the manual that converts from the more normal encoding.)

# Zielwerte pro Knoten
y = torch.tensor([
    [10.0],
    [20.0],
    [30.0]
])

# Graphobjekt
data = Data(x=x, edge_index=edge_index, y=y)

# --------------------------------------------------
# GNN Modell
# --------------------------------------------------

class TrafficGNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Graph convolution
        self.conv1 = GCNConv(2, 16)

        # zweite Graph convolution
        self.conv2 = GCNConv(16, 1)

        self.relu = nn.ReLU()

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        # Nachrichtenaustausch über Graph
        x = self.conv1(x, edge_index)

        x = self.relu(x)

        x = self.conv2(x, edge_index)

        return x

model = TrafficGNN()

# --------------------------------------------------
# Training
# --------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_function = nn.MSELoss()

for epoch in range(200):

    prediction = model(data)

    loss = loss_function(prediction, data.y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, loss.item())