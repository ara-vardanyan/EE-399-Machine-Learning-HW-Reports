# Experimenting with Curve Fitting

**Author**:  

Ara Vardanyan

**Abstract**:

In this assignment, we will analyze a sample dataset using various mathematical models. Our goal is to fit these models to the data using least-squares error and explore the minima found in different combinations of parameter tuning by generating 2D loss landscapes. We will also experiment with different models such as line, parabola, and 19th-degree polynomial and compare their performance on different splits of the data. Through these exercises, we aim to gain a deeper understanding of the strengths and weaknesses of different models and techniques for machine learning.

## Introduction

We will begin by fitting a model of the form `f(x) = A*cos(Bx) + Cx + D` to the data using least-squares error. This technique allows us to find the optimal values for the parameters A, B, C, and D that minimize the error between the model predictions and the observed data. Next, we will fix two of these parameters and sweep through values of the other two to generate a 2D loss landscape. This will allow us to visualize how the error changes as we vary different combinations of parameters.

In addition to fitting this specific model to the data, we will also experiment with other models such as a line, parabola, and 19th-degree polynomial. We will split the data into training and test sets and fit these models to the training data. We will then compute the least-square error for each model on both the training and test data. This will allow us to compare how well different models fit the data. This will be done once more with a different training data split which will allow us to examine how the different models behave with different data splits.

Through this assignment, we aim to gain a deeper understanding of how different models fit with data. By experimenting with different models and data splits, we hope to develop a better understanding of the models strengths and weaknesses, how they behave with different data splits, and learn how to choose the best model for a given dataset and data split.

## Theoretical Background

One of the key concepts that underlies the process of machine learning is the error or loss function. A loss function is a mathematical function that quantifies how well a model fits a given dataset. The loss function measures the difference between the predicted values of the model and the observed values in the data. The goal of fitting a model to data is to find the values for the model’s parameters that minimize the loss function.

The specific loss function chosen for this assignment was least-quares error. The least-squares error is defined as the sum of the squared differences between the observed values and the predicted values of a model. Mathematically, this can be expressed as:

E = (1/n) * ∑(f(x<sub>j</sub>) - y<sub>j</sub>)<sup>2</sup> for j = 1 to n


where E is the least-squares error, n is the number of data points, f(x<sub>j</sub>) is the predicted value of the model for the j-th data point, and y<sub>j</sub> is the observed value for the j-th data point.

To find the optimal values for the parameters of a model that minimize the least-squares error, we can use various optimization techniques such as gradient descent or Newton's method. These techniques allow us to iteratively adjust the values of the parameters until we find a set of values that minimize the least-squares error.

In addition to least-squares error, another important concept in this report is the idea of a loss or error landscape. A loss landscape is a visualization of how the error changes as we vary different combinations of parameters. By generating a 2D loss landscape, we can see how the error changes as we sweep through values of two parameters while keeping the other two fixed. This can help us identify any minima present in the loss landscape and gain insights into how different combinations of parameter values affect the performance of our model.

## Algorithm Implementation and Development

Initialization of sample data:

```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
N = len(X)
```

### Problem I: Calculating Minimum Error and Determining Parameters

A Python function is defined for the given model `f(x) = A*cos(Bx) + Cx + D`
```
def func(x, A, B, C, D):
    return A * np.cos(B * x) + C * x + D
```

The model is then fit to the sample dataset.
```
params, params_covariance = scipy.optimize.curve_fit(func, X, Y)
```

The least squares-error is then determined by first calculating the fitted values x<sub>j</sub>, then applying the least-squares error formula:

E = (1/n) * ∑(f(x<sub>j</sub>) - y<sub>j</sub>)<sup>2</sup> for j = 1 to n

```
fitted_values = func(X, *params)
least_squares_error = np.sum((fitted_values - Y) ** 2) / N
```


### Problem II: Generating 2D loss landscapes

Next, we fix two parameters and sweep through values of the other two parameters to see the loss at each combination of unfixed parameter values, generating a 2D loss landscape. This is done with the following function that accepts 2 fixed parameters and the values to be swept through of the other two parameters, and returns a loss landscape.

```
def sweep(param1_name, param1_values, param2_name, param2_values, fixed_params):
    error_landscape = np.zeros((len(param1_values), len(param2_values)))

    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            fitted_values = func(X, *fixed_params(p1, p2))
            mse = np.sum((fitted_values - Y) ** 2) / N
            error_landscape[i, j] = mse

    return error_landscape
```

We then generate the landscape for each possible combination of 2 unfixed and 2 swept parameters.

```
A_values = np.linspace(-10, 10, 100)
B_values = np.linspace(0, 2 * np.pi, 100)
C_values = np.linspace(-1, 1, 100)
D_values = np.linspace(20, 60, 100)

error_landscapes = [
    ('C', C_values, 'D', D_values, lambda p1, p2: (params[0], params[1], p1, p2)),
    ('B', B_values, 'D', D_values, lambda p1, p2: (params[0], p1, params[2], p2)),
    ('B', B_values, 'C', C_values, lambda p1, p2: (params[0], p1, p2, params[3])),
    ('A', A_values, 'D', D_values, lambda p1, p2: (p1, params[1], params[2], p2)),
    ('A', A_values, 'C', C_values, lambda p1, p2: (p1, params[1], p2, params[3])),
    ('A', A_values, 'B', B_values, lambda p1, p2: (p1, p2, params[2], params[3])),
]
```

The loss landscapes are then plotted.

```
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

for i, (ax, (param1_name, param1_values, param2_name, param2_values, fixed_params)) in enumerate(zip(axes.flat, error_landscapes)):
    error_landscape = sweep(param1_name, param1_values, param2_name, param2_values, fixed_params)
    pc = ax.pcolor(param2_values, param1_values, error_landscape, shading='auto', cmap='viridis')
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title(f'Error Landscape for {param1_name} and {param2_name}')

fig.colorbar(pc, ax=axes.ravel().tolist(), label='Mean Squared Error')
plt.show()
```

### Problem III & IV: Line, Parabola, and 19th Degree Polynomial Model Fits

To examine how different data splits impact model performace, we test three models with two data splits. We first use the first 20 data points as training data and the last 10 data points as the testing data.

```
# Split data into training and test sets
X_train, Y_train = X[:20], Y[:20]
X_test, Y_test = X[20:], Y[20:]
```

Then we fit each model to the training data.

```
linear_Y_train = linear_fit(X_train)
parabola_Y_train = parabola_fit(X_train)
poly19_Y_train = poly19_fit(X_train)

linear_Y_test = linear_fit(X_test)
parabola_Y_test = parabola_fit(X_test)
poly19_Y_test = poly19_fit(X_test)
```

Next, we calculate the least-squares error for each model on both the training and testing data sets.

```
def least_squares_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

models = ["linear", "parabola", "19th-degree polynomial"]
train_errors = [
    least_squares_error(Y_train, linear_Y_train),
    least_squares_error(Y_train, parabola_Y_train),
    least_squares_error(Y_train, poly19_Y_train),
]
test_errors = [
    least_squares_error(Y_test, linear_Y_test),
    least_squares_error(Y_test, parabola_Y_test),
    least_squares_error(Y_test, poly19_Y_test),
]

# Print the errors
print("Least-squares errors on training data:")
for model, error in zip(models, train_errors):
    print(f"{model}: {error:.4f}")

print("\nLeast-squares errors on test data:")
for model, error in zip(models, test_errors):
    print(f"{model}: {error:.4f}")
```

We do the exact same thing again, this time with the first 10 and last 10 data points as training data and the middle 10 data points as testing data. Besides the splitting of the data, the implementation remains the same.

```
# Split data into training and test sets
X_train = np.concatenate((X[:10], X[-10:]))
Y_train = np.concatenate((Y[:10], Y[-10:]))
X_test, Y_test = X[10:20], Y[10:20]
```

## Computational Results

### Problem I: Calculating Minimum Error and Determining Parameters

#### Model
`f(x) = A*cos(Bx) + Cx + D`

#### Loss Function

Least squares-error:

E = (1/n) * ∑(f(x<sub>j</sub>) - y<sub>j</sub>)<sup>2</sup> for j = 1 to n

#### Results

- Least-squares error: 2.53678
- Parameters
  - A: 2.1717
  - B: 0.9093
  - C: 0.7325
  - D: 31.4528
 


### Problem II: Generating 2D loss landscapes

#### Loss Landscapes

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework1/figures/ErrorLandscapeAllCombos.png'>
</p>

#### Top 5 Local Minima

| Rank | Local Minima for C and D | Local Minima for B and D | Local Minima for B and C | Local Minima for A and D | Local Minima for A and C | Local Minima for A and B |
|------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 1    | C=0.74, D=31.31, Error=2.54 | B=0.89, D=31.31, Error=2.94 | B=0.89, C=0.74, Error=2.95 | A=2.12, D=31.31, Error=2.56 | A=2.12, C=0.74, Error=2.55 | A=1.92, B=0.89, Error=2.92 |
| 2    | C=0.72, D=31.72, Error=2.56 | B=5.39, D=31.31, Error=2.94 | B=5.39, C=0.74, Error=2.95 | A=2.12, D=31.72, Error=2.61 | A=2.12, C=0.72, Error=2.61 | A=1.92, B=5.39, Error=2.92 |
| 3    | C=0.76, D=30.91, Error=2.62 | B=0.89, D=31.72, Error=3.02 | B=0.89, C=0.72, Error=2.97 | A=2.12, D=30.91, Error=2.84 | A=2.12, C=0.76, Error=2.73 | A=-1.92, B=5.52, Error=3.14 |
| 4    | C=0.70, D=32.12, Error=2.66 | B=5.39, D=31.72, Error=3.02 | B=5.39, C=0.72, Error=2.97 | A=2.12, D=32.12, Error=2.98 | A=2.12, C=0.70, Error=2.92 | A=-1.92, B=0.76, Error=3.14 |
| 5    | C=0.78, D=30.51, Error=2.77 | B=0.89, D=30.91, Error=3.19 | B=0.89, C=0.76, Error=3.18 | A=2.32, D=30.51, Error=3.44 | A=2.12, C=0.78, Error=3.16 | A=1.52, B=0.95, Error=3.64 |

### Problem III & IV: Fitting Different Models to Training and Test Data

#### Training Data: First 20 Data Points, Testing Data: Last 10 Data Points 


| Model                 | Least-squares Error on Training Data | Least-squares Error on Test Data      |
|-----------------------|--------------------------------------|---------------------------------------|
| Linear                | 5.0299                               | 11.3141                               |
| Parabola              | 4.5179                               | 75.9277                               |
| 19th-degree Polynomial| 0.0000                               | 3.8454 x 10<sup>22</sup>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework1/figures/ExtrapolatedModelFits.png'>
</p>


#### Training Data: First 10 and last 10 Data Points, Testing Data: Middle 10 Data Points 



| Model                 | Least-squares Error on Training Data | Least-squares Error on Test Data      |
|-----------------------|--------------------------------------|---------------------------------------|
| Linear                | 3.4287                               | 8.6454                               |
| Parabola              | 3.4256                               | 8.4437                               |
| 19th-degree Polynomial| 0.0000                               | 1.8093 x 10<sup>5</sup>

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework1/figures/InterpolatedModelFits.png'>
</p>


## Summary and Conclusions

### Problem III & IV

The first thing to note from the results in this problem is the difficulty the 19th degree polynomial has with the testing set. It has 0 loss on the training sets for both data splits but an extremely large loss on both testing sets. This can be attributed to 'polynomial wiggle'. The high degree of the polynomial causes it to overfit to the training data which renders the model useless for making predictions on data it has not seen. This is a big problem as the goal of machine learning is to create models that generalize well.

Splitting the data with the first 20 data points as the training data and the last 10 data points as the testing data tests the models' abilities to extrapolate data. Splitting the data with the first 10 and last 10 data points as the training set and the middle 10 data points as the testing set tests the models' abilities to interpolate data. All models had a lower loss on both the training and testing set for the interpolation task. From this we can conclude that these models perform interpolation better than extrapolation. Extrapolation is a more difficult task in most cases as it gives less context or information as to what the next value could be so this behavior from the model is probable and as such, it is expected.

Finally, we can seee that the parabola performed worse than the linear model with the extrapolation task but slightly better with the interpolation. The nature of a 2nd degree polynomial is to bend once. I believe that since the polynomial had the context of the last 10 data points, the fit for the interpolation chose a better point for it to bend to minimize the loss leading to the results we mentioned. The data set was roughly linear so it makes sense that the polynomial struggles with extrapolating this.
