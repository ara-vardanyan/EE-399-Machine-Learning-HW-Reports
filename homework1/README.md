# Experimenting with Curve Fitting

**Author**:  

Ara Vardanyan

**Abstract**:

In this assignment, we fit various models to a dataset for the purpose of experimentation and discovering the flaws in different types of models. We fit these models using minimization of least-squares error to optimize the parameters. We explore the many minima found in different combinations of parameter tuning by generating 2d error landscapes. We also fit line, parabola, and 19th-degree polynomial models to the data with different splits of the dataset and discuss how and why the results may differ.


## Introduction

In this assignment, we will explore the process of analyzing data using various mathematical models and techniques. Given a dataset, we will fit a model of the form f(x) = A*cos(Bx) + Cx + D to the data using least-squares error. We will then write code to find the minimum error and determine the optimal values for the parameters A, B, C, and D. Next, we will fix two of these parameters and sweep through values of the other two to generate a 2D loss landscape. We will visualize the results using pcolor and identify the number of minima present as we sweep through the parameters.

In addition to fitting this specific model to the data, we will also experiment with other models such as line, parabola, and 19th-degree polynomial. We will split the data into training and test sets and fit these models to the training data. We will then compute the least-square error for each model on both the training and test data. Finally, we will repeat this process using a different split of the data into training and test sets and compare the results.


## Problem I: Fitting a Model with Least-Squares Error

### Part (i): Minimizing Error and Determining Parameters

In this part, a custom Python function is implemented to fit the given model `f(x) = A*cos(Bx) + Cx + D` to the provided dataset using the least-squares error. The parameters `A`, `B`, `C`, and `D` are determined by minimizing the error.

**Results:**

- Least-squares error: 2.53678
- Parameters
  - A: 2.1717
  - B: 0.9093
  - C: 0.7325
  - D: 31.4528

### Part (ii): Generating 2D Loss Landscapes

Here, the error landscape is visualized in 2D by fixing two parameters and sweeping through values of the other two parameters. This is done for all combinations of fixed and swept parameters. The resulting landscapes are displayed using `pcolor`.

**Results:**

- With C and D fixed and sweeping through 100 values for both A and B, 4 minima were found with an error of 2.92. The corresponding parameters were:
  - A: 1.92, B: 0.89, Error: 2.92
  - A: 1.92, B: 5.39, Error: 2.92
  - A: 2.12, B: 0.89, Error: 2.92
  - A: 2.12, B: 5.39, Error: 2.92  

<p align='center'>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/582e1a1eb501f1eff2d2bf364d59e76e6c26b80c/homework1/figures/ErrorLandscapeAB.png'>
</p>
  
- Below is the error landscape for all combinations of 2 fixed parameters and 2 swept parameters.
<p align='center'>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/3260b07660433576dd95ca9d5b5475cda75480d3/homework1/figures/ErrorLandscapeAllCombos.png' width = "900" >
</p>

## Problem III: Fitting Different Models to Training and Test Data

### Part (iii): Fitting Line, Parabola, and 19th Degree Polynomial to First 20 Data Points

In this section, a line, parabola, and 19th degree polynomial are fit to the first 20 data points (training data). The least-squares error is computed for each model over the training points, as well as over the remaining 10 data points (test data).

**Results:**

- Least-squares error on training data:
  - linear: 5.0299
  - parabola: 4.5179
  - 19th-degree polynomial: 0.0000
- Least-squares error on test data:
  - linear: 11.3141
  - parabola: 75.9277
  - 19th-degree polynomial: 38454308324079644442624.0000



<p align='center'>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework1/figures/ExtrapolatedModelFits.png'>
</p>

### Part (iv): Fitting Models to First 10 and Last 10 Data Points

The same models are fit to the first 10 and last 10 data points as training data. The least-squares error is computed for each model on the test data, which consists of the 10 held out middle data points.

**Results:**

- Least-squares error on training data:
  - linear: 3.4287
  - parabola: 3.4256
  - 19th-degree polynomial: 0.0000
- Least-squares error on test data:
  - linear: 8.6454
  - parabola: 8.4437
  - 19th-degree polynomial: 180926.4988

<p align='center'>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework1/figures/InterpolatedModelFits.png'>
</p>

## Comparison of Results


(Include a brief comparison of the results obtained in parts (iii) and (iv) and any insights or observations.)

## Conclusion

This report summarizes the process and results of fitting various models to the provided dataset using least-squares error minimization. The models are evaluated on their training and test data performance, and the results are compared to gain insights into the behavior of the different models.
