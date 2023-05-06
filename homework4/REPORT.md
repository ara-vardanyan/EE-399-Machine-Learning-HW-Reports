# Neural Network Analysis and Model Comparison on Interpolation and Extrapolation Tasks and MNIST Data Set

**Author**:

Ara Vardanyan

**Abstract**:

This report explores the application of neural networks on interpolation vs extrapolation tasks as well as on classifying digits in the MNIST data set. We train neural networks on interpolation and extrapolation tasks and evaluate and compare their performance. We also train a neural network on the task of classifying digits in the MNIST data set with dimensionality reduction techniques in order to evaluate the performance and compare it with other classifiers such as LSTM, SVM (support vector machines), and decision trees.

---

## Introduction

This report aims to investigate the performance of neural networks on two primary tasks: interpolation vs. extrapolation and digit classification using the MNIST data set.

In the first task, we focus on fitting a three-layer feedforward neural network (FNN) to a given data set, following different training and testing strategies to analyze the model's performance in interpolation and extrapolation tasks. We compute the least-squares error to evaluate the model's accuracy and compare it with the models from homework one.

The second task involves working with the MNIST data set, a popular benchmark for image classification algorithms. We employ Principal Component Analysis (PCA) to reduce the data set's dimensionality before training a feedforward neural network to classify the digits. We compare the performance of the neural network with other classification algorithms, such as Long Short-Term Memory (LSTM) networks, Support Vector Machines (SVM), and decision trees.

By exploring neural networks in these two tasks, we gather insights into their strengths and limitations which further our understanding of architecture selection and development for different tasks and use-cases.

---

## Theoretical Background

### Feedforward Neural Networks (FNNs):
Feedforward neural networks (FNNs) are a type of artificial neural network that consist of multiple layers of interconnected nodes or neurons. These networks are characterized by the flow of information in a single direction, from the input layer through one or more hidden layers and finally to the output layer. FNNs are widely used in supervised learning tasks, such as regression and classification, due to their ability to model complex, nonlinear relationships between input and output variables.

### Least-squares Error:
Least-squares error is a measure of the difference between the predicted values produced by a model and the actual values observed in the data. In the context of this assignment, it is used to evaluate the performance of the neural network models by quantifying the difference between their predictions and the actual values in both the training and test data. Lower least-squares errors indicate better model performance. Mathematically, least-squares error can be expressed as:

E = (1/n) * ∑(f(x<sub>j</sub>) - y<sub>j</sub>)<sup>2</sup> for j = 1 to n

where E is the least-squares error, n is the number of data points, f(x<sub>j</sub>) is the predicted value of the model for the j-th data point, and y<sub>j</sub> is the observed value for the j-th data point.

### Principal Component Analysis (PCA):
PCA is a technique for reducing the dimensionality of a dataset by projecting it onto a lower-dimensional space. It is closely related to SVD, as the principal components can be derived from the V matrix in the SVD decomposition. The goal is to retain as much variance in the data as possible while reducing the number of dimensions.

### Support Vector Machines (SVM):
SVM is a supervised machine learning algorithm that can be used for classification or regression tasks. It aims to find the optimal hyperplane that best separates the data points of different classes. The optimal hyperplane is determined by maximizing the margin between the closest data points of different classes, which are called support vectors. SVM can handle linearly separable as well as non-linearly separable data using kernel functions.

### Decision Trees:
Decision Trees are a type of supervised machine learning algorithm used for classification and regression tasks. The algorithm recursively splits the input space into regions based on feature values to create a tree structure. Each node in the tree represents a decision based on a feature value, while the leaf nodes represent the predicted class labels. Decision Trees can handle both categorical and numerical data and are easy to interpret, but they are prone to overfitting if not properly pruned or limited in depth.

---

## Algorithm Implementation and Development

Importing necessary libraries:

```
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
```

### Problem 1: Interpolation and Extrapolation tasks

Initialization of sample data:

```
X = np.arange(0,31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

Construction of three layer feed forward neural network:

```
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_network(model, X_train, Y_train, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = Variable(X_train)
        targets = Variable(Y_train)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

def evaluate_least_squares_loss(model, X_train, Y_train, X_test, Y_test):
    criterion = nn.MSELoss()

    with torch.no_grad():
        train_outputs = model(X_train)
        train_mse = criterion(train_outputs, Y_train)
        test_outputs = model(X_test)
        test_mse = criterion(test_outputs, Y_test)

    train_least_squares_error = torch.sqrt(train_mse).item()
    test_least_squares_error = torch.sqrt(test_mse).item()

    return train_least_squares_error, test_least_squares_error

def compute_accuracy(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    return 1 - mae.item()

```

Preparing data for extrapolation task:

```
# Normalizing the data
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X = X / 31
Y = Y / 55

# Creating the training and test sets
X_train_extrap = X[:20]
Y_train_extrap = Y[:20]
X_test_extrap = X[20:]
Y_test_extrap = Y[20:]

# Converting the data to pytorch tensors
X_train_tensor_extrap = torch.FloatTensor(X_train_extrap)
Y_train_tensor_extrap = torch.FloatTensor(Y_train_extrap)
X_test_tensor_extrap = torch.FloatTensor(X_test_extrap)
Y_test_tensor_extrap = torch.FloatTensor(Y_test_extrap)
```

Training the neural network for extrapolation task:

```
input_size = 1
hidden_size = 64
output_size = 1
learning_rate = 0.01
epochs = 1000

model = ThreeLayerNN(input_size, hidden_size, output_size)

train_network(model, X_train_tensor_extrap, Y_train_tensor_extrap, epochs, learning_rate)
```

Evaluating the loss and accuracy on extrapolation task:

```
# Evaluate the loss and accuracy
train_loss, test_loss = evaluate_least_squares_loss(model, X_train_tensor_extrap, Y_train_tensor_extrap, X_test_tensor_extrap, Y_test_tensor_extrap)
train_accuracy = compute_accuracy(Y_train_tensor_extrap, model(X_train_tensor_extrap))
test_accuracy = compute_accuracy(Y_test_tensor_extrap, model(X_test_tensor_extrap))

print(f'Training Loss (Extrapolation): {round(train_loss, 4)}')
print(f'Test Loss (Extrapolation): {round(test_loss, 4)}')
print(f'Training Accuracy (Extrapolation): {round(train_accuracy * 100, 4)}%')
print(f'Test Accuracy (Extrapolation): {round(test_accuracy * 100, 4)}%')
```

Preparing data for interpolation task:

```
# Creating the training and test sets
X_train_interp = np.concatenate((X[:10], X[-10:]))
Y_train_interp = np.concatenate((Y[:10], Y[-10:]))
X_test_interp = X[10:20]
Y_test_interp = Y[10:20]

# Converting the data to pytorch tensors
X_train_tensor_interp = torch.FloatTensor(X_train_interp)
Y_train_tensor_interp = torch.FloatTensor(Y_train_interp)
X_test_tensor_interp = torch.FloatTensor(X_test_interp)
Y_test_tensor_interp = torch.FloatTensor(Y_test_interp)
```

Training the neural network for interpolation task:

```
model_interp = ThreeLayerNN(input_size, hidden_size, output_size)

train_network(model_interp, X_train_tensor_interp, Y_train_tensor_interp, epochs, learning_rate)
```

Evaluating the loss and accuracy on interpolation task:

```
train_loss_interp, test_loss_interp = evaluate_least_squares_loss(model_interp, X_train_tensor_interp, Y_train_tensor_interp, X_test_tensor_interp, Y_test_tensor_interp)
train_accuracy_interp = compute_accuracy(Y_train_tensor_interp, model_interp(X_train_tensor_interp))
test_accuracy_interp = compute_accuracy(Y_test_interp_tensor, model_interp(X_test_tensor_interp))

print(f'Training Loss (Interpolation): {round(train_loss_interp, 4)}')
print(f'Test Loss (Interpolation): {round(test_loss_interp, 4)}')
print(f'Training Accuracy (Interpolation): {round(train_accuracy_interp * 100, 4)}%')
print(f'Test Accuracy (Interpolation): {round(test_accuracy_interp * 100, 4)}%')
```

### Problem 2: Classifying digits of MNIST data set

Creating the feed forward neural network:

```
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return correct / total

def train_network(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = Variable(inputs)
            targets = Variable(targets)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

def evaluate_loss_accuracy(model, data_loader):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = Variable(inputs)
            targets = Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total += targets.size(0)
            correct += (torch.max(outputs, 1)[1] == targets).sum().item()

    return total_loss / len(data_loader), correct / total
```

Loading the MNIST dataset:

```
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

X_train = train_dataset.data.numpy()
X_test = test_dataset.data.numpy()

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
```

Computing the first 20 PCA modes of the digit images:

```
pca = PCA(n_components=20)
pca.fit(X_train_flat)
X_train_pca = pca.transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)
```

Classifying the digits with the neural network:

```
# Convert the PCA data to tensors
X_train_pca_tensor = torch.FloatTensor(X_train_pca)
X_test_pca_tensor = torch.FloatTensor(X_test_pca)
Y_train_tensor = torch.LongTensor(train_dataset.targets)
Y_test_tensor = torch.LongTensor(test_dataset.targets)

train_pca_dataset = torch.utils.data.TensorDataset(X_train_pca_tensor, Y_train_tensor)
test_pca_dataset = torch.utils.data.TensorDataset(X_test_pca_tensor, Y_test_tensor)
train_pca_loader = torch.utils.data.DataLoader(train_pca_dataset, batch_size=100, shuffle=True)
test_pca_loader = torch.utils.data.DataLoader(test_pca_dataset, batch_size=100, shuffle=False)

input_size = 20
hidden_size = 128
output_size = 10
learning_rate = 0.001
epochs = 50

# Create and train the model
model = FFNN(input_size, hidden_size, output_size)

train_network(model, train_pca_loader, epochs, learning_rate)
```

Evaluating the loss and accuracy of the FFNN on digit classification:

```
train_loss, train_accuracy = evaluate_loss_accuracy(model, train_pca_loader)
test_loss, test_accuracy = evaluate_loss_accuracy(model, test_pca_loader)

print(f'Training Loss: {round(train_loss, 4)}')
print(f'Test Loss: {round(test_loss, 4)}')
print(f'Training Accuracy: {round(train_accuracy * 100, 4)}%')
print(f'Test Accuracy: {round(test_accuracy * 100, 4)}%')
```


---

## Computational Results

### Extrapolation and Interpolation:

| Task          | Metric        | Training | Testing |
|---------------|---------------|----------|---------|
| Extrapolation | Loss          | 0.0263   | 0.1827  |
| Interpolation | Loss          | 0.0298   | 0.0553  |
| Interpolation | Accuracy      | 97.3836% | 95.0551%|
| Extrapolation | Accuracy      | 97.9064% | 82.5943%|


### MNIST Data Set Digit Classification:

| Classifier          | Train Set Accuracy | Test Set Accuracy | Training Loss | Testing Loss |
|---------------------|--------------------|-------------------|---------------|--------------|
| Neural Network      | 96.78%             | 96.27%            | 0.1844        | 0.2654       |
| SVM                 | 91.03%             | 90.65%            |               |              |
| Decision Tree       | 100.00%            | 82.88%            |               |              |

The performance of SVM and decision trees were retrieved from the report for [Homework 3](../homework3/REPORT.md) (where 102 PCA modes were used).

---

## Summary and Conclusions

### Extrapolation and Interpolation:

In this report we first investigated the performance of neural networks on the tasks of interpolation vs extrapolation. We trained two 3 layer feed forward neural network models. Both models used the same exact architecture and only differed in their data split. The first model's data was split with the first 20 data points as the training data and the last 10 data points as the testing data which tests the model's ability to extrapolate data. The second model's data was split with the first 10 and last 10 data points as the training set and the middle 10 data points as the testing set which tests the model's ability to interpolate data.

While both models had a similar training accuracy, the model that was trained to interpolate achieved both a significantly higher test set accuracy (95.06% vs 82.59%) and lower loss (0.0553 vs 0.1827) than the model trained to extrapolate. From this we can conclude that neural network models perform interpolation better than extrapolation. Extrapolation is a more difficult task in most cases as it gives less context or information as to what the next value could be so this behavior from the models was expected, especially since we saw similar differences in performances with different models in [Homework 1](../homework1/REPORT.md).

The neural network model strongly outperformed the linear, parabolic, and 19th degree polynomial fits from [Homework 1](../homework1/REPORT.md) on both interpolation and extrapolation. For the interpolation task, the parabolic fit model from [Homework 1](../homework1/REPORT.md) had the lowest least-squares error on the test set of 8.44 vs 0.0553 for the neural network. For the extrapolation task, the linear fit model from [Homework 1](../homework1/REPORT.md) had the lowest least-squares error on the test set of 11.3141 vs 0.1827 for the neural network. From this evidence, we can conlude that neural networks are more well versed in interpolation and extrapolation tasks than linear, parabolic, or polynomial fits.

### MNIST Data Set Digit Classification:

Our next exploration in this report was the classification of digits using neural networks. We took the popular MNIST benchmarking data set and employed Principal Component Analysis (PCA) to reduce the data set's dimensionality to the top 20 PCA modes. We then trained and tested a feed forward neural network on the reduced dimensionality data set.

We observed an accuracy of 96.27% on the test set with the neural network model. This is a significantly better performance than the SVM's 90.65% and the decision tree's 82.88% accuracy on the test set in [Homework 3](../homework3/REPORT.md). It is important to note that the neural network was only trained on the top 20 PCA modes while the decision tree and SVM were trained on the top 102 PCA modes. This pushes us to the conclusion that neural networks may be much better suited for image classification tasks than SVMs and decison trees.