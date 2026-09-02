import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# Trainingsdaten erzeugen
# --------------------------------------------------

# X = Eingabevektoren
# Beispiel:
#   Nachfrage in 3 Zonen
#
# Shape:
#   [anzahl_samples, anzahl_features]

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.0],
    [3.0, 1.0, 2.0],
    [0.0, 2.0, 1.0]
])

# Y = Zielwerte
# Beispiel:
#   Stauindikatoren auf 2 Kanten

Y = torch.tensor([
    [10.0, 20.0],
    [ 5.0,  7.0],
    [13.0, 15.0],
    [ 4.0,  8.0]
])

# --------------------------------------------------
# Modell definieren
# --------------------------------------------------

class TrafficNet(nn.Module):
    # ("TrafficNet extends nn.Module")

    def __init__(self):
        super().__init__()

        # Linear layer:
        # 3 inputs -> 16 hidden neurons
        self.hidden = nn.Linear(3, 16)

        # 16 hidden neurons -> 2 outputs
        self.output = nn.Linear(16, 2)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):

        # hidden layer
        x = self.hidden(x)

        # nonlinearity
        x = self.relu(x)

        # output layer
        x = self.output(x)

        return x

# Modell erzeugen
model = TrafficNet()

# --------------------------------------------------
# Loss Function
# --------------------------------------------------

# Mean squared error
loss_function = nn.MSELoss()

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(model.parameters(), lr=0.01)
# ("lr" stands for "learning rate")

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(1000):

    # Vorhersage
    predictions = model(X)

    # Fehler berechnen
    loss = loss_function(predictions, Y)

    # Alte Gradienten löschen
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Gewichte updaten
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item()}")

# --------------------------------------------------
# Test
# --------------------------------------------------

test_input = torch.tensor([[1.5, 1.5, 2.0]])

prediction = model(test_input)

print("Test input, Prediction:")
print(test_input)
print(prediction)