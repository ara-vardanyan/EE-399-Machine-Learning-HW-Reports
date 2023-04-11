# EE 399 Spring Quarter 2023: Homework 1 Report

## Instructor: J. Nathan Kutz
**Submitted by**: Ara Vardanyan

## Introduction

This report documents the process and results of the first homework assignment for EE 399 Spring Quarter 2023, under the instruction of Dr. J. Nathan Kutz. The assignment involves analyzing a given dataset and fitting different models to it using least-squares error minimization.

## Problem I: Fitting a Model with Least-Squares Error

### Part (i): Minimizing Error and Determining Parameters

In this part, a custom Python function is implemented to fit the given model `f(x) = A*cos(Bx) + Cx + D` to the provided dataset using the least-squares error. The parameters `A`, `B`, `C`, and `D` are determined by minimizing the error.

**Results:**

- Least-squares error: 2.53678
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
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/582e1a1eb501f1eff2d2bf364d59e76e6c26b80c/homework1/figures/ErrorLandscapeAllCombos.png' width = "500" >
</p>

## Problem III: Fitting Different Models to Training and Test Data

### Part (iii): Fitting Line, Parabola, and 19th Degree Polynomial to First 20 Data Points

In this section, a line, parabola, and 19th degree polynomial are fit to the first 20 data points (training data). The least-squares error is computed for each model over the training points, as well as over the remaining 10 data points (test data).

**Results:**

- Least-squares error for line, parabola, and 19th degree polynomial on training data: [Values]
- Least-squares error for line, parabola, and 19th degree polynomial on test data: [Values]

### Part (iv): Fitting Models to First 10 and Last 10 Data Points

The same models are fit to the first 10 and last 10 data points as training data. The least-squares error is computed for each model on the test data, which consists of the 10 held out middle data points.

**Results:**

- Least-squares error for line, parabola, and 19th degree polynomial on test data: [Values]

## Comparison of Results

(Include a brief comparison of the results obtained in parts (iii) and (iv) and any insights or observations.)

## Conclusion

This report summarizes the process and results of fitting various models to the provided dataset using least-squares error minimization. The models are evaluated on their training and test data performance, and the results are compared to gain insights into the behavior of the different models.
