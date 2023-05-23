# Analyzing the Performance of a SHRED Model on Sea-Surface Temperature Data with Respect to Time Lag, Noise, and Number of Sensors

**Author**:

Ara Vardanyan

**Abstract**:

In this report, we train a SHallow REcurrent Decoder (SHRED) model on NOAA's sea-surface temperature data and assess its performance. Key variables in this analysis include time lag, the level of Gaussian noise, and the number of sensors. By manipulating these parameters, we aim to understand their impact on the model's predictive accuracy, providing insights into the robustness and scalability of the SHRED model under various data conditions.

---

## Introduction

This assignment dives into the exploration of the SHallow REcurrent Decoder (SHRED) model's performance in reconstructing high-dimensional spatio-temporal fields. Our study specifically focuses on the model's ability to predict sea-surface temperatures, utilizing the NOAA Optimum Interpolation SST V2 dataset.

SHRED, which merges an LSTM network with a shallow decoder network (SDN), will be trained to make predictions from a trajectory of sensor measurements over different time lags. We aim to evaluate how the model performs under varying conditions such as changes in time lag, addition of Gaussian noise to the data, and variations in the number of sensors used.

---

## Theoretical Background

### Long Short-Term Memory (LSTM):
LSTM networks, part of the recurrent neural network (RNN) family, are designed to process sequential data while overcoming the vanishing gradient problem, a common issue in RNNs. Unlike standard RNNs, LSTMs incorporate gating mechanisms in their architecture, which enable them to selectively remember or forget information over long sequences. This ability to learn long-term dependencies in the data makes them powerful tools for modeling complex sequences and temporal dynamics.

### Shallow Decoder Networks (SDNs):
SDNs are a type of feedforward neural network designed to handle high-dimensional output. Contrasting with deeper networks, SDNs have fewer layers which can make them more interpretable and easier to optimize. Despite their simplicity, SDNs can learn complex mappings from their inputs to high-dimensional outputs, especially when paired with other types of networks that process the input data.

### SHallow REcurrent Decoder (SHRED):
The SHRED model is a hybrid network architecture that combines the strengths of LSTM networks and SDNs. This architecture enables SHRED to handle high-dimensional spatio-temporal fields. The model first uses an LSTM network to capture temporal dependencies in the data. The output of the LSTM network is then processed by an SDN to reconstruct the spatio-temporal field. This combination enables the model to infer high-dimensional fields from sequences of measurements over time, making it particularly useful for tasks where the goal is to reconstruct a field from sensor data.

---

## Algorithm Implementation and Development

We first randomly select 3 sensor locations and set the trajectory length (lags) to 52, corresponding to one year of measurements.
```
import numpy as np
from processdata import load_data
from processdata import TimeSeriesDataset
import models
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```

We now select indices to divide the data into training, validation, and test sets.
```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```

sklearn's MinMaxScaler is used to preprocess the data for training and we generate input/output pairs for the training, validation, and test sets. 
```
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```

We train the model using the training and validation datasets.
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```

Finally, we generate reconstructions from the test set and print mean square error compared to the ground truth.
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
```

We train models for various values of time lag and evaluate the performace as a function of time lag.
```
# Set the desired time lag values
lag_values = [26, 52, 78, 104, 130]

# Load the data
load_X = load_data('SST')

# Define other parameters
num_sensors = 3
load_size = load_X.shape[0]
sensor_locations = np.random.choice(load_X.shape[1], size=num_sensors, replace=False)
sc = MinMaxScaler()
sc = sc.fit(load_X[:, sensor_locations])  # Use only the selected sensor locations

# Initialize lists to store the performance results
mse_values = []

for lag in lag_values:
    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((load_size - lag, lag, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = sc.transform(load_X[i:i+lag, sensor_locations])

    # Generate datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)
    data_out = torch.tensor(sc.transform(load_X[lag:, sensor_locations]), dtype=torch.float32).to(device)
    dataset = TimeSeriesDataset(data_in, data_out)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the SHRED model
    shred = models.SHRED(num_sensors, num_sensors, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset.dataset, valid_dataset.dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=False)

    # Evaluate the model on the test set
    test_recons = sc.inverse_transform(shred(dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    mse_values.append(mse)

# Plot the performance as a function of the time lag variable
plt.plot(lag_values, mse_values, marker='o')
plt.xlabel('Time Lag')
plt.ylabel('MSE')
plt.title('Performance as a Function of Time Lag')
plt.show()
```

We train models for different values of noise variance and evaluate the performace as a function of noise level.
```
# Set the desired noise variance levels
noise_variances = [0.01, 0.05, 0.1, 0.2, 0.5]

# Load the data
load_X = load_data('SST')

# Define other parameters
num_sensors = 3
load_size = load_X.shape[0]
sensor_locations = np.random.choice(load_X.shape[1], size=num_sensors, replace=False)

# Initialize lists to store the performance results
mse_values = []

for noise_variance in noise_variances:
    # Generate noisy data
    noisy_load_X = load_X.copy()
    for i in range(load_X.shape[1]):
        noise = np.random.normal(0, noise_variance, load_X.shape[0])
        noisy_load_X[:, i] += noise

    # Scale the noisy data
    sc = MinMaxScaler()
    sc = sc.fit(noisy_load_X[:, sensor_locations])

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((load_size - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = sc.transform(noisy_load_X[i:i+lags, sensor_locations])

    # Generate datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)
    data_out = torch.tensor(sc.transform(noisy_load_X[lags:, sensor_locations]), dtype=torch.float32).to(device)
    dataset = TimeSeriesDataset(data_in, data_out)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the SHRED model
    shred = models.SHRED(num_sensors, num_sensors, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset.dataset, valid_dataset.dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=False)

    # Evaluate the model on the test set
    test_recons = sc.inverse_transform(shred(dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    mse_values.append(mse)

# Plot the performance as a function of the noise level
plt.plot(noise_variances, mse_values, marker='o')
plt.xlabel('Noise Variance')
plt.ylabel('MSE')
plt.title('Performance as a Function of Noise Level')
plt.show()
```

We train models for different numbers of sensors (1-5) and evaluate the performace as a function of the number of sensors.
```
# Set the desired number of sensors
num_sensors_values = [1, 2, 3, 4, 5]

# Load the data
load_X = load_data('SST')

# Define other parameters
lags = 52
load_size = load_X.shape[0]
sensor_locations = np.arange(load_X.shape[1])  # All sensor locations

# Initialize lists to store the performance results
mse_values = []

for num_sensors in num_sensors_values:
    # Randomly select sensor locations
    selected_sensor_locations = np.random.choice(sensor_locations, size=num_sensors, replace=False)

    # Scale the data
    sc = MinMaxScaler()
    sc = sc.fit(load_X[:, selected_sensor_locations])

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((load_size - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = sc.transform(load_X[i:i+lags, selected_sensor_locations])

    # Generate datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_in = torch.tensor(all_data_in, dtype=torch.float32).to(device)
    data_out = torch.tensor(sc.transform(load_X[lags:, selected_sensor_locations]), dtype=torch.float32).to(device)
    dataset = TimeSeriesDataset(data_in, data_out)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Train the SHRED model
    shred = models.SHRED(num_sensors, num_sensors, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    models.fit(shred, train_dataset.dataset, valid_dataset.dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=False)

    # Evaluate the model on the test set
    test_recons = sc.inverse_transform(shred(dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    mse_values.append(mse)

# Plot the performance as a function of the number of sensors
plt.plot(num_sensors_values, mse_values, marker='o')
plt.xlabel('Number of Sensors')
plt.ylabel('MSE')
plt.title('Performance as a Function of Number of Sensors')
plt.show()
```

---

## Computational Results

### Performance of SHRED as a function of time lag

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/1ba345647b22cc65f6f51a3ca68ee31c296e09fb/homework6/figures/SHREDPerformanceTimeLag.png'>
</p>

### Performance of SHRED as a function of noise level

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/1ba345647b22cc65f6f51a3ca68ee31c296e09fb/homework6/figures/SHREDPerformanceNoiseLevel.png'>
</p>

### Performance of SHRED as a function of number of sensors

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/1ba345647b22cc65f6f51a3ca68ee31c296e09fb/homework6/figures/SHREDPerformanceNumberOfSensors.png'>
</p>


---

## Summary and Conclusions

In this report, we have successfully trained and evaluated the SHallow REcurrent Decoder (SHRED) model using sea-surface temperature data obtained from NOAA's Optimum Interpolation SST V2 dataset. The SHRED model, which combines an LSTM network with a shallow decoder network, is designed to reconstruct high-dimensional spatio-temporal fields from a trajectory of sensor measurements.

Our work involved training the SHRED model and examining its performance under varying conditions. The parameters we manipulated included the time lag variable, Gaussian noise levels in the data, and the number of sensors.

Based on our results, the time lag does have an impact on the performance of the SHRED model, as seen by the variation in Mean Squared Error (MSE) values. For lags of 26, 52, and 104, the model achieved relatively low MSE values (0.0245, 0.0255, and 0.0234 respectively), indicating that it can accurately predict the spatio-temporal data within these intervals.

Interestingly, for lags of 78 and 130, the MSE jumped significantly to 0.240, implying a decrease in the model's performance. This suggests that the model's capability to infer the high-dimensional fields from sequences of measurements tends to diminish at these larger time lags.

The impact of noise level on the SHRED model's performance is quite clear from the obtained results. As the noise level increases, the Mean Squared Error (MSE) value generally increases, indicating a degradation in the model's predictive performance.

For low noise levels of 0.01, 0.05, and 0.1, the model maintains a relatively consistent and low MSE value (between 0.017 and 0.018). This suggests that the model is quite robust to low levels of noise, maintaining accurate prediction capabilities in spite of minor perturbations in the data.

However, when the noise level increases to 0.2 and 0.5, the MSE jumps significantly to 0.205 and 0.315, respectively. This large increase in error signifies that the model's performance degrades considerably under these higher levels of noise.

The influence of the number of sensors on the performance of the SHRED model is evident from the observed results. As the number of sensors increases, the Mean Squared Error (MSE) value generally decreases, indicating an improvement in the model's predictive performance.

For a single sensor, the model recorded a fairly high MSE of 0.10. As we added more sensors, the MSE significantly decreased, registering 0.06 for two sensors, and further decreasing to 0.02 for three and four sensors. This suggests that additional sensor input greatly enhances the model's ability to accurately predict the spatio-temporal data, possibly due to the additional information and coverage provided by multiple sensor readings.

However, when we increased the number of sensors to five, the MSE slightly increased to 0.03. This could be due to the model's diminishing return from the increased complexity and redundancy of the data as the number of sensors increases beyond a certain point.

Therefore, we conclude that while increasing the number of sensors generally improves the performance of the SHRED model, care must be taken not to oversaturate the model with too many sensors, which could lead to minor performance degradation.

In conclusion, our investigation has provided valuable insights into the performance characteristics of the SHRED model under varying conditions. From the impact of the time lag, the influence of noise level, to the effect of sensor numbers, our findings enhance our understanding of this LSTM-based model's behavior in reconstructing high-dimensional spatio-temporal fields.

The observations from this study underscore the complexity of predicting spatio-temporal data and the challenges that arise from the variations in time lag, noise level, and sensor numbers. They highlight the need for careful selection and tuning of these parameters to achieve optimal performance.