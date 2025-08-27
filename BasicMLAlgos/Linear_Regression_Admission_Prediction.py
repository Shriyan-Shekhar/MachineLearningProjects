import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.linear_model import LinearRegression

class myModel (nn.Module):
    def __init__ (self):
        super (myModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear (7,7)
        self.fc2 = nn.Linear (7,7)
        self.fc3 = nn.Linear (7,7)
        self.fc4 = nn.Linear (7,1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class LinearRegressionModel (nn.Module):
    def __init__ (self):
        super (LinearRegressionModel, self).__init__()
        self.model = LinearRegression()
    def fit(self,x,y):
        self.model.fit (x,y)
    def evaluate (self, x):
        return self.model.predict(x)


data = pd.read_csv('Admission_Predict.csv')
df = pd.DataFrame(data)
df.columns = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit']
df = df.drop(['Serial No.'], axis=1)
x = df [['CGPA']]
x = df.drop(['Chance of Admit'], axis=1)
y = df ['Chance of Admit']

correlation_matrix = df.corr()

print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix')
plt.show()

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split (x, y,test_size = 0.1, train_size = 0.9)


model2 = LinearRegressionModel()
model2.fit (x_train, y_train)
predictions = model2.evaluate(x_test)


#hyperparameters
epochs = 10000
learning_rate = 0.005
batch_size = len(x_train)

model = myModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

x_train_tensor = torch.tensor (x_train, dtype = torch.float32)
y_train_tensor = torch.tensor (y_train.to_numpy(), dtype = torch.float32)

train_data = TensorDataset (x_train_tensor, y_train_tensor)
train_loader = DataLoader (dataset = train_data, batch_size = batch_size, shuffle = False)

for epoch in range (epochs):
    model.train()
    all_train_outputs = []
    optimizer.zero_grad()
    outputs = model (x_train_tensor)
    loss = criterion (outputs.squeeze(), y_train_tensor) #Learning: Remember to squeeze the output as MSE loss will lead to inaccurate values due to shape mismatch (it only gives warning without squeeze but extremely wrong values)
    loss.backward()
    optimizer.step()
    all_train_outputs.append(outputs)
    print (f"Epoch {epoch+1} Done, Loss = {loss.item()}")

all_train_outputs= torch.cat(all_train_outputs).detach().numpy()

x_test_tensor = torch.tensor (x_test, dtype = torch.float32)
y_test_tensor = torch.tensor (y_test.to_numpy(), dtype = torch.float32)

test_data = TensorDataset (x_test_tensor, y_test_tensor)
test_loader = DataLoader (dataset = test_data, batch_size = batch_size, shuffle = False)

with torch.no_grad():
    total = 0
    correct = 0
    all_outputs = []
    for x, y in test_loader:
        outputs = model (x)
        all_outputs.append(outputs)
        for i in range (len(outputs)):
            total += 1
            if abs (outputs[i] - y[i]) < 0.1:
                correct += 1
    all_outputs = torch.cat(all_outputs).numpy()
    print (f"Accuracy = {100 * correct / total}%")

plt.scatter (x_test[:,5], y_test_tensor.numpy(), label = 'Actual', color = 'red')
plt.scatter (x_test[:,5], predictions, label = 'Predicted SKLearn', color = 'blue')
plt.scatter (x_test[:,5], all_outputs, label = 'Predicted Pytorch', color = 'green')
#plt.scatter (x_test, y_test, label = 'Actual', color = 'red')
#plt.scatter (x_test, all_outputs, label = 'Predicted', color = 'blue')
plt.xlabel ('Input Features')
plt.ylabel ('Chance of Admit')
plt.title ('Actual vs Predicted')
plt.legend ()
plt.show ()
