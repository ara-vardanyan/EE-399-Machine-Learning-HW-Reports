# Predicting Lorenz System Behavior with Feed Forward, Long Short Term Memory, Recurrent, and Echo State Networks

**Author**:

Ara Vardanyan

**Abstract**:


---

## Introduction



---

## Theoretical Background

### Feedforward Neural Networks (FFNNs):
Feedforward neural networks (FNNs) are a type of artificial neural network that consist of multiple layers of interconnected nodes or neurons. These networks are characterized by the flow of information in a single direction, from the input layer through one or more hidden layers and finally to the output layer. FNNs are widely used in supervised learning tasks, such as regression and classification, due to their ability to model complex, nonlinear relationships between input and output variables.


---

## Algorithm Implementation and Development

Importing necessary libraries:

```
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
import tensorflow_addons as tfa
import tensorflow as tf
```

Generating Lorenz system data:
```
def lorenz_deriv(x_y_z, t0, sigma=10, beta=8/3, rho=28):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_data(rho, seed=123):
    dt = 0.01
    T = 8
    t = np.arange(0, T + dt, dt)

    np.random.seed(seed)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])

    nn_input = np.zeros((100 * (len(t) - 1), 3))
    nn_output = np.zeros_like(nn_input)

    for j in range(100):
        nn_input[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, :-1, :]
        nn_output[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, 1:, :]

    return nn_input, nn_output
```

### Functions for creating and training models

#### Feed Forward Neural Network (FFNN)
```
def build_ffnn(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_shape[0], activation='linear')) # Output layer
    model.compile(optimizer='adam', loss='mse')
    return model

def train_ffnn(model, inputs, targets, epochs=100):
    model.fit(inputs, targets, epochs=epochs, verbose=1)
```

#### Long Short Term Memory Network (LSTM)
```
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(input_shape[1], activation='linear')) # Output layer
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(model, inputs, targets, epochs=100):
    model.fit(inputs, targets, epochs=epochs, verbose=1)
```

#### Recurrent Neural Network (RNN)
```
def build_rnn(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', input_shape=input_shape))
    model.add(Dense(input_shape[1], activation='linear')) # Output layer
    model.compile(optimizer='adam', loss='mse')
    return model

def train_rnn(model, inputs, targets, epochs=100):
    model.fit(inputs, targets, epochs=epochs, verbose=1)
```

#### Echo State Network (ESN)
```
def create_esn(input_shape, units, connectivity=0.1, leaky=1, spectral_radius=0.9):
    inputs = tf.keras.Input(shape=input_shape)
    esn_outputs = tfa.layers.ESN(units, connectivity, leaky, spectral_radius)(inputs)
    output = tf.keras.layers.Dense(3)(esn_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_esn(X_train, y_train, X_test, y_test, input_shape, reservoir_size, epochs=50, batch_size=32):
    esn_model = create_esn(input_shape, reservoir_size)
    esn_history = esn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)
    return esn_model, esn_history
```

### Training models

Preparing data to train models:
```
# Generate data for each rho value
inputs_10, targets_10 = generate_data(10)
inputs_28, targets_28 = generate_data(28)
inputs_40, targets_40 = generate_data(40)

# Concatenate the inputs and targets
inputs = np.concatenate([inputs_10, inputs_28, inputs_40])
targets = np.concatenate([targets_10, targets_28, targets_40])
```

#### Feed Forward Neural Network (FFNN)
```
# Build the model
ffnn = build_ffnn(inputs.shape[1:])

# Train the model
train_ffnn(ffnn, inputs, targets, epochs=25)
```

Reshape inputs for LSTM, RNN, and ESN [samples, time steps, features]
```
inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))
```

#### Long Short Term Memory Network (LSTM)
```
# Build the model
lstm = build_lstm(inputs.shape[1:])

# Train the model
train_lstm(lstm, inputs, targets, epochs=25)
```

#### Recurrent Neural Network (RNN)
```
# Build the model
rnn = build_rnn(inputs.shape[1:])

# Train the model
train_rnn(rnn, inputs, targets, epochs=25)
```

#### Echo State Network (ESN)
```
# Split data into training and test sets
train_size = int(0.8 * len(inputs))
X_train, X_test = inputs[:train_size], inputs[train_size:]
y_train, y_test = targets[:train_size], targets[train_size:]

# Define ESN parameters
input_shape = inputs.shape[1:]
reservoir_size = 64
epochs = 25
batch_size = 32

# Train the ESN model
esn_model, esn_history = train_esn(X_train, y_train, X_test, y_test, input_shape, reservoir_size, epochs, batch_size)
```

### Predicting Lorenz System for rho = 17, 35

Defining function for plotting predictions:
```
def plot_predictions(new_targets, predictions, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(new_targets[:, 0], new_targets[:, 1], new_targets[:, 2], 'r', label='Actual')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], 'b', label='Predicted')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
```

#### Predicting with FFNN
```
# Generate new data for rho=17
rho_17 = 17
inputs_17, targets_17 = generate_data(rho_17)

# Use the trained model to make predictions for the new data
predictions_17 = ffnn.predict(inputs_17)

mse_17 = mean_squared_error(targets_17, predictions_17)
print(f'FFNN Mean Squared Error on data with rho={rho_17}: {mse_17}')

plot_predictions(targets_17, predictions_17, f'FFNN for rho={rho_17}')

# Generate new data for rho=35
rho_35 = 35
inputs_35, targets_35 = generate_data(rho_35)

# Use the trained model to make predictions for the new data
predictions_35 = ffnn.predict(inputs_35)

mse_35 = mean_squared_error(targets_35, predictions_35)
print(f'FFNN Mean Squared Error on data with rho={rho_35}: {mse_35}')

plot_predictions(targets_35, predictions_35, f'FFNN for rho={rho_35}')
```

### Predicting with LSTM
```
# Generate new data for rho=17
rho_17 = 17
inputs_17, targets_17 = generate_data(rho_17)

# Reshape new_inputs for LSTM [samples, time steps, features]
inputs_17 = inputs_17.reshape((inputs_17.shape[0], 1, inputs_17.shape[1]))

# Use the trained model to make predictions for the new data
predictions_17 = lstm.predict(inputs_17)

# Calculate and print the MSE
mse_17 = mean_squared_error(targets_17, predictions_17)
print(f'LSTM Mean Squared Error on data with rho={rho_17}: {mse_17}')

plot_predictions(targets_17, predictions_17, f'LSTM for rho={rho_17}')

# Generate new data for rho=35
rho_35 = 35
inputs_35, targets_35 = generate_data(rho_35)

# Reshape new_inputs for LSTM [samples, time steps, features]
inputs_35 = inputs_35.reshape((inputs_35.shape[0], 1, inputs_35.shape[1]))

# Use the trained model to make predictions for the new data
predictions_35 = lstm.predict(inputs_35)

# Calculate and print the MSE
mse_35 = mean_squared_error(targets_35, predictions_35)
print(f'LSTM Mean Squared Error on data with rho={rho_35}: {mse_35}')

plot_predictions(targets_35, predictions_35, f'LSTM for rho={rho_35}')
```

### Predicting with RNN
```
# Generate new data for rho=17
rho_17 = 17
inputs_17, targets_17 = generate_data(rho_17)

# Reshape new_inputs for RNN [samples, time steps, features]
inputs_17 = inputs_17.reshape((inputs_17.shape[0], 1, inputs_17.shape[1]))

# Use the trained model to make predictions for the new data
predictions_17 = rnn.predict(inputs_17)

# Calculate and print the MSE
mse_17 = mean_squared_error(targets_17, predictions_17)
print(f'RNN Mean Squared Error on new data with rho={rho_17}: {mse_17}')

plot_predictions(targets_17, predictions_17, f'RNN for rho={rho_17}')

# Generate new data for rho=35
rho_35 = 35
inputs_35, targets_35 = generate_data(rho_35)

# Reshape new_inputs for RNN [samples, time steps, features]
inputs_35 = inputs_35.reshape((inputs_35.shape[0], 1, inputs_35.shape[1]))

# Use the trained model to make predictions for the new data
predictions_35 = rnn.predict(inputs_35)

# Calculate and print the MSE
mse_35 = mean_squared_error(targets_35, predictions_35)
print(f'RNN Mean Squared Error on new data with rho={rho_35}: {mse_35}')

plot_predictions(targets_35, predictions_35, f'RNN for rho={rho_35}')
```

### Predicting with ESN
```
# Generate new data for rho=17
rho_17 = 17
inputs_17, targets_17 = generate_data(rho_17)

# Reshape new_inputs for ESN [samples, time steps, features]
inputs_17 = inputs_17.reshape((inputs_17.shape[0], 1, inputs_17.shape[1]))

# Use the trained model to make predictions for the new data
predictions_17 = esn_model.predict(inputs_17)

# Calculate and print the MSE
mse_17 = mean_squared_error(targets_17, predictions_17)
print(f'ESN Mean Squared Error on new data with rho={rho_17}: {mse_17}')

plot_predictions(targets_17, predictions_17, f'ESN for rho={rho_17}')

# Generate new data for rho=35
rho_35 = 35
inputs_35, targets_35 = generate_data(rho_35)

# Reshape new_inputs for ESN [samples, time steps, features]
inputs_35 = inputs_35.reshape((inputs_35.shape[0], 1, inputs_35.shape[1]))

# Use the trained model to make predictions for the new data
predictions_35 = esn_model.predict(inputs_35)

# Calculate and print the MSE
mse_35 = mean_squared_error(targets_35, predictions_35)
print(f'ESN Mean Squared Error on new data with rho={rho_35}: {mse_35}')

plot_predictions(targets_35, predictions_35, f'ESN for rho={rho_35}')
```

---

## Computational Results

### Error on Lorenz system with unseen values of rho

Epochs: 25

For rho=17:

| Model | Mean Squared Error |
|-------|-------------------|
| RNN   | 0.006688 |
| LSTM  | 0.007858 |
| FFNN  | 0.009169 |
| ESN   | 7.348020 |

For rho=35:

| Model | Mean Squared Error |
|-------|-------------------|
| LSTM  | 0.012208 |
| RNN   | 0.014535 |
| FFNN  | 0.018447 |
| ESN   | 9.366474 |

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/ffnn_predictions_rho_17.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/ffnn_predictions_rho_35.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/lstm_predictions_rho_17.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/lstm_predictions_rho_35.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/rnn_predictions_rho_17.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/rnn_predictions_rho_35.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/esn_predictions_rho_17.png'>
</p>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/686628772fe8867e813148ff1d39f0c95545d441/homework5/figures/esn_predictions_rho_35.png'>
</p>


---

## Summary and Conclusions

### Extrapolation and Interpolation:

In this report we first investigated the performance of neural networks on the tasks of interpolation vs extrapolation. We trained two 3 layer feed forward neural network models. Both models used the same exact architecture and only differed in their data split. The first model's data was split with the first 20 data points as the training data and the last 10 data points as the testing data which tests the model's ability to extrapolate data. The second model's data was split with the first 10 and last 10 data points as the training set and the middle 10 data points as the testing set which tests the model's ability to interpolate data.

While both models had a similar training accuracy, the model that was trained to interpolate achieved both a significantly higher test set accuracy (95.06% vs 82.59%) and lower loss (0.0553 vs 0.1827) than the model trained to extrapolate. From this we can conclude that neural network models perform interpolation better than extrapolation. Extrapolation is a more difficult task in most cases as it gives less context or information as to what the next value could be so this behavior from the models was expected, especially since we saw similar differences in performances with different models in [Homework 1](../homework1/REPORT.md).

The neural network model strongly outperformed the linear, parabolic, and 19th degree polynomial fits from [Homework 1](../homework1/REPORT.md) on both interpolation and extrapolation. For the interpolation task, the parabolic fit model from [Homework 1](../homework1/REPORT.md) had the lowest least-squares error on the test set of 8.44 vs 0.0553 for the neural network. For the extrapolation task, the linear fit model from [Homework 1](../homework1/REPORT.md) had the lowest least-squares error on the test set of 11.3141 vs 0.1827 for the neural network. From this evidence, we can conlude that neural networks are more well versed in interpolation and extrapolation tasks than linear, parabolic, or polynomial fits.

### MNIST Data Set Digit Classification:

Our next exploration in this report was the classification of digits using neural networks. We took the popular MNIST benchmarking data set and employed Principal Component Analysis (PCA) to reduce the data set's dimensionality to the top 20 PCA modes. We then trained and tested a feed forward neural network on the reduced dimensionality data set.

We observed an accuracy of 96.27% on the test set with the neural network model. This is a significantly better performance than the SVM's 90.65% and the decision tree's 82.88% accuracy on the test set in [Homework 3](../homework3/REPORT.md). It is important to note that the neural network was only trained on the top 20 PCA modes while the decision tree and SVM were trained on the top 102 PCA modes. This pushes us to the conclusion that neural networks may be much better suited for image classification tasks than SVMs and decison trees.