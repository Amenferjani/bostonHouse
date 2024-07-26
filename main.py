import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
from model import Model

torch.manual_seed(50)

data = pd.read_csv(
    "c:/Users/AMEN/Desktop/pytorch/bostonHouse/BostonHousing.csv")
data = data.dropna()  


dataFrame = data.drop("medv", axis=1)
target = data["medv"]

xTrain, xTest, yTrain, yTest = train_test_split(
    dataFrame, target, test_size=0.2, random_state=42)

xTrain = torch.FloatTensor(xTrain.values)
xTest = torch.FloatTensor(xTest.values)
yTrain = torch.FloatTensor(yTrain.values)
yTest = torch.FloatTensor(yTest.values)

# Initialize the model, loss function, and optimizer
model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# Training parameters
epochs = 10000
patience = 5
best_loss = float('inf')
epochs_without_improvement = 0
best_model_weights = copy.deepcopy(model.state_dict())

# Training loop with early stopping
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    yPred = model(xTrain)
    loss = criterion(yPred, yTrain.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        model.eval()
        with torch.no_grad():
            yEval = model(xTest)
            val_loss = criterion(yEval, yTest.unsqueeze(1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Load the best model weights
model.load_state_dict(best_model_weights)

# Save the model
# torch.save(model.state_dict(),"houseModel.pt")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    yEval = model(xTest)
    test_loss = criterion(yEval, yTest.unsqueeze(1))
    print("Evaluation Loss:", test_loss.item())

# Calculate evaluation metrics
mse = mean_squared_error(yTest, yEval.squeeze())
mae = mean_absolute_error(yTest, yEval.squeeze())
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Predict and evaluate individual test samples
print("\n ****************** \n")
total_error = 0
with torch.no_grad():
    for i in range(len(xTest)):
        yVal = model(xTest[i])
        error = abs(yVal.item() - yTest[i].item())
        total_error += error
        print(
            f"Predicted: {yVal.item():.2f}, Actual: {yTest[i].item():.2f}, Error: {error:.2f}")

average_error = total_error / len(xTest)
print("Average Error:", average_error)
