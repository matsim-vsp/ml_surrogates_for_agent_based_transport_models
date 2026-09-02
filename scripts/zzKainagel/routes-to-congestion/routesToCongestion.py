import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ------------------------------------------------------------
# 1. GRAPH (Straßennetz)
# ------------------------------------------------------------

edge_index = torch.tensor([
    [0, 1, 2, 3, 1, 2],
    [1, 2, 3, 0, 3, 0]
], dtype=torch.long)
# (recall that this means edges "from top to bottom"). Ie.e. we have 6 edges.

num_nodes = 4
num_edges = edge_index.shape[1]

# ------------------------------------------------------------
# 2. ROUTES statt OD
# ------------------------------------------------------------
# Jede Route ist eine Sequenz von Edge-Indizes

routes = [
    [0, 1, 2],   # route A
    [4, 3],      # route B
    [5]          # route C
]
# (along the 6 edges)

route_flows = torch.tensor([10.0, 5.0, 3.0])
# ((OD) flows along each route)

# ------------------------------------------------------------
# 3. Edge flow aggregation (klassischer Schritt in Traffic Models)
# ------------------------------------------------------------

edge_flow = torch.zeros(num_edges)

for route, flow in zip(routes, route_flows):
    for edge in route:
        edge_flow[edge] += flow

# (edges sind nummeriert 0..5)

# ------------------------------------------------------------
# 4. Node features (minimal)
# ------------------------------------------------------------

print("edge_flow=", edge_flow)

x = torch.ones((num_nodes, 2))

print("x=", x)

x[:, 0] = edge_flow[edge_index[0]].mean(dim=0)
x[:, 1] = edge_flow[edge_index[1]].mean(dim=0)
# (wenn ich es richtig verstehe, dann werden die edge flows als incoming und outgoing an die jeweiligen Knoten drangehängt)

print("x=", x)

# ausprogrammiert mit den obigen Werten:
x[:,0] = edge_flow[ [0,1,2,3,1,2] ].mean( dim=0 )
x[:,1] = edge_flow[ [1,2,3,0,3,0] ].mean( dim=0 )

# M.E.:
# x[0:0] = edge_flow[0].mean(dim=0)
# ...
# x[5:0] = edge_flow[2].mean(dim=0)
# x[0:1] = edge_flow[1].mean(dim=0)
# ...
# x[5:1] = edge_flow[0].mean(dim=0)

# dim=0 = Mittelung entlang der 0ten Dimension = Mittelwerte der Spalten.  Reduziert irgendwie den 6dim array auf 4dim.  Ich
# vermute, dass es über die jeweils "gleichen" Indizes mittelt.  Also z.B. alle flows, die fromNode=1 haben, werden gemittelt und an
# node 1 als outgoing drangehängt. :-(

# ------------------------------------------------------------
# 5. Target congestion (synthetic)
# ------------------------------------------------------------

y = (edge_flow + torch.tensor([2.0, 1.0, 3.0, 1.5, 2.5, 1.0])).view(-1, 1)
#( ich denke mir, das das die edge flows als congestion nimmt, und "2,1,3,..." draufaddiert. )

# ------------------------------------------------------------
# 6. Data object
# ------------------------------------------------------------

data = Data(
    x=x,
    edge_index=edge_index,
    edge_flow=edge_flow,
    y=y
)

# ------------------------------------------------------------
# 7. Model
# ------------------------------------------------------------

class RouteGNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 16)

        self.edge_mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        # --------------------------------------------------------
        # GNN propagation
        # --------------------------------------------------------
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # --------------------------------------------------------
        # edge representation
        # --------------------------------------------------------
        src, dst = edge_index
        edge_feat = torch.cat([x[src], x[dst]], dim=1)

        # optional: route-informed feature
        edge_feat = edge_feat + data.edge_flow.unsqueeze(1)

        out = self.edge_mlp(edge_feat)

        return out

# ------------------------------------------------------------
# 8. Training
# ------------------------------------------------------------

model = RouteGNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):

    pred = model(data)

    loss = loss_fn(pred, data.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, loss.item())

print(data.edge_flow )
print( model(data) )