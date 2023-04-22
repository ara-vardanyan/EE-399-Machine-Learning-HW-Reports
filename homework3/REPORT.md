# Analysis and Classification of the MNIST Dataset using SVD, LDA, SVM, and Decision Trees

**Author**:

Ara Vardanyan

**Abstract**:

This report explores the application of Singular Value Decomposition (SVD), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees on the MNIST dataset. We first analyze dimension reductionality of data with SVD, then the performance of each of these classifiers on digit classification. This investigation serves as a comprehensive comparison of different classification techniques, offering insights into their distinct capabilities and performance metrics.

---

## Introduction

The MNIST dataset is a popular and widely used benchmark in the field of machine learning and pattern recognition, containing 70,000 handwritten digit images. In this report, we present an analysis of the MNIST dataset using Singular Value Decomposition (SVD), visualization of the dataset in a reduced feature space, and evaluation of multiple classifiers to identify and classify individual digits in the dataset. We begin by performing SVD analysis and examining the singular value spectrum to determine the necessary number of modes for image reconstruction. We also interpret the U, Σ, and V matrices obtained from the SVD and project the data onto a 3D plot using selected V-modes.

Following the initial analysis, we explore the performance of three different classifiers: Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees. We first build an LDA classifier to identify two and three chosen digits, then determine the most difficult and easiest pairs of digits to separate using the LDA classifier. Next, we evaluate the classification performance of SVM and Decision Trees on all ten digits. Finally, we compare the performance of LDA, SVM, and Decision Trees on the hardest and easiest pairs of digits to separate. The analysis includes a discussion of classifier performance on both the training and test sets, providing a comprehensive evaluation of the methods applied.

---

## Theoretical Background

### Singular Value Decomposition (SVD):
SVD is a linear algebra technique used to decompose a matrix into three other matrices: U, Σ, and V. It has applications in dimensionality reduction, data compression, and noise reduction. In the context of the MNIST dataset, SVD allows us to capture the most significant features of the images, thereby reducing the complexity of the problem and computation while preserving important information.

### Principal Component Analysis (PCA):
PCA is a technique for reducing the dimensionality of a dataset by projecting it onto a lower-dimensional space. It is closely related to SVD, as the principal components can be derived from the V matrix in the SVD decomposition. The goal is to retain as much variance in the data as possible while reducing the number of dimensions.

### Linear Discriminant Analysis (LDA):
LDA is a supervised linear dimensionality reduction and classification technique. It seeks to maximize the separation between different classes by projecting the data onto a lower-dimensional space. LDA aims to minimize the within-class variance while maximizing the between-class variance, resulting in improved classification performance.

### Support Vector Machines (SVM):
SVM is a supervised machine learning algorithm that can be used for classification or regression tasks. It aims to find the optimal hyperplane that best separates the data points of different classes. The optimal hyperplane is determined by maximizing the margin between the closest data points of different classes, which are called support vectors. SVM can handle linearly separable as well as non-linearly separable data using kernel functions.

### Decision Trees:
Decision Trees are a type of supervised machine learning algorithm used for classification and regression tasks. The algorithm recursively splits the input space into regions based on feature values to create a tree structure. Each node in the tree represents a decision based on a feature value, while the leaf nodes represent the predicted class labels. Decision Trees can handle both categorical and numerical data and are easy to interpret, but they are prone to overfitting if not properly pruned or limited in depth.

---

## Algorithm Implementation and Development

Importing necessary libraries
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import TruncatedSVD
```

Loading the MNIST data set
```
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist['data'].to_numpy(), mnist['target'].to_numpy()
```

### SVD analysis of the digit images

Reshape each image into a column vector
```
X_reshaped = X.reshape(X.shape[0], -1)
```

Perform SVD decomposition
```
U, S, Vt = np.linalg.svd(X_reshaped, full_matrices=False)
```

Plotting the top 9 most important SVD modes
```
num_modes = 0
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    eigen_digit = Vt[i].reshape(28, 28)
    ax.imshow(eigen_digit)
    ax.set_title(f'SVD Mode {i + 1}')
    ax.axis('off')
    
plt.tight_layout()
plt.show()
```

### Analyzing singular value spectrum and determining rank for good image reconstruction

```
# Singular Value Spectrum
plt.plot(S)
plt.xlabel("Modes")
plt.ylabel("Singular Values")
plt.title("Singular Value Spectrum")
plt.show()

# Cumulative Variance Explained
explained_variance_ratio = S ** 2 / np.sum(S ** 2)
cumulative_variance_explained = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_variance_explained)
plt.xlabel("Modes")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained")
plt.show()

# Determine the number of modes necessary for good image reconstruction
desired_variance = 0.95
rank_r = np.argmax(cumulative_variance_explained >= desired_variance) + 1
print(f"Rank r for 95% variance: {rank_r}")
```

### Projecting digits onto 3 V-modes on a 3D plot

Reconstructing the reduced dimensionality representation of the data set
```
# Setting number of components as enough for 95% variance
n_components = 102

U_reduced = U[:, :n_components]
S_reduced = np.diag(S[:n_components])
Vt_reduced = Vt[:n_components, :]

X_reduced = U_reduced @ S_reduced
print("X_reduced shape:", X_reduced.shape)
```

Plotting projection of digits onto 3 V-modes on 3D plot
```
from mpl_toolkits.mplot3d import Axes3D

# Select the V-modes (columns) to be plotted
selected_columns = [1, 2, 4]
X_selected = X_reduced[:, selected_columns]

Y_int = Y.astype(int)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Use a colormap to map the digit labels (0-9) to colors
colors = plt.cm.get_cmap('tab10', 10)

for i in range(10):
    indices = Y_int == i
    ax.scatter(X_selected[indices, 0], X_selected[indices, 1], X_selected[indices, 2], color=colors(i), label=str(i), alpha=0.6, edgecolors='w')

# Set axes labels with padding
ax.set_xlabel('V-mode 2', labelpad=10)
ax.set_ylabel('V-mode 3', labelpad=10)
ax.set_zlabel('V-mode 5', labelpad=10)

# Add title and configure legend
ax.set_title('3D Scatter Plot of Selected V-modes')
ax.legend(title="Digits", loc='upper center', ncol=10)

plt.show()
```

### LDA classifier

Splitting the data set into training and testing data
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_reduced, Y, test_size=0.3, random_state=42)
```

Implementing train and test functions for LDA classifier
```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def train_lda_classifier(X_train, Y_train, digits):
    # Filter the training data to only include specified digits
    mask = np.isin(Y_train, digits)
    X_filtered = X_train[mask]
    Y_filtered = Y_train[mask]

    # Train the LDA classifier
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_filtered, Y_filtered)

    return lda

def test_lda_classifier(lda, X_test, Y_test, digits):
    # Filter the test data to only include specified digits
    mask = np.isin(Y_test, digits)
    X_filtered = X_test[mask]
    Y_filtered = Y_test[mask]

    # Predict the labels using the LDA classifier
    Y_pred = lda.predict(X_filtered)

    # Calculate the accuracy
    accuracy = np.mean(Y_pred == Y_filtered)

    return accuracy
```

Training and testing LDA on 2 digits
```
digits_2 = ['0', '2']
lda_classifier_2_digits = train_lda_classifier(X_train, Y_train, digits_2)
test_accuracy_2_digits = test_lda_classifier(lda_classifier_2_digits, X_test, Y_test, digits_2)
train_accuracy_2_digits = test_lda_classifier(lda_classifier_2_digits, X_train, Y_train, digits_2)

print(f"LDA classifier accuracy on test set of digits {digits_2}: {test_accuracy_2_digits * 100:.2f}%")
print(f"LDA classifier accuracy on train set of digits {digits_2}: {train_accuracy_2_digits * 100:.2f}%")
```


Training and testing LDA on 3 digits
```
digits_3 = ['1', '3', '5']
lda_classifier_3_digits = train_lda_classifier(X_train, Y_train, digits_3)
test_accuracy_3_digits = test_lda_classifier(lda_classifier_3_digits, X_test, Y_test, digits_3)
train_accuracy_3_digits = test_lda_classifier(lda_classifier_3_digits, X_train, Y_train, digits_3)

print(f"LDA classifier accuracy on test sets of digits {digits_3}: {test_accuracy_3_digits * 100:.2f}%")
print(f"LDA classifier accuracy on train sets of digits {digits_3}: {train_accuracy_3_digits * 100:.2f}%")
```

Training and testing LDA on each unique pair of digits to find the easiest and hardest pairs of digits to separate
```
from itertools import combinations

unique_digits = np.unique(Y)
digit_pairs = list(combinations(unique_digits, 2))

test_accuracies = []
train_accuracies = []

for pair in digit_pairs:
    lda_classifier_pair = train_lda_classifier(X_train, Y_train, pair)

    test_accuracy_pair = test_lda_classifier(lda_classifier_pair, X_test, Y_test, pair)
    test_accuracies.append(test_accuracy_pair)
    
    train_accuracy_pair = test_lda_classifier(lda_classifier_pair, X_train, Y_train, pair)
    train_accuracies.append(train_accuracy_pair)

    print(f"LDA classifier accuracy on test set for digits {pair}: {test_accuracy_pair * 100:.2f}%")
    print(f"LDA classifier accuracy on train set for digits {pair}: {train_accuracy_pair * 100:.2f}%")

max_test_accuracy_index = np.argmax(test_accuracies)
min_test_accuracy_index = np.argmin(test_accuracies)

max_train_accuracy_index = np.argmax(train_accuracies)
min_train_accuracy_index = np.argmin(train_accuracies)

easiest_test_digits = digit_pairs[max_test_accuracy_index]
hardest_test_digits = digit_pairs[min_test_accuracy_index]

easiest_train_digits = digit_pairs[max_train_accuracy_index]
hardest_train_digits = digit_pairs[min_train_accuracy_index]

print(f"\nThe easiest digits to separate on the test set are {easiest_test_digits} with an accuracy of {test_accuracies[max_test_accuracy_index] * 100:.2f}%")
print(f"The hardest digits to separate on the test set are {hardest_test_digits} with an accuracy of {test_accuracies[min_test_accuracy_index] * 100:.2f}%")

print(f"\nThe easiest digits to separate on the train set are {easiest_train_digits} with an accuracy of {train_accuracies[max_train_accuracy_index] * 100:.2f}%")
print(f"The hardest digits to separate on the train set are {hardest_train_digits} with an accuracy of {train_accuracies[min_train_accuracy_index] * 100:.2f}%")
```

### SVM and Decision Tree classifiers

Implementing train and test functions for SVM and Decision Tree classifiers
```
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_svm_classifier(X_train, Y_train, kernel='linear', C=1):
    svm = LinearSVC(C=C, dual=False, multi_class='ovr', random_state=42)
    svm.fit(X_train, Y_train)
    return svm

def test_svm_classifier(svm, X_test, Y_test):
    Y_pred = svm.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy

def train_decision_tree_classifier(X_train, Y_train, criterion='gini', max_depth=None):
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dt.fit(X_train, Y_train)
    return dt

def test_decision_tree_classifier(dt, X_test, Y_test):
    Y_pred = dt.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy
```


Training and testing classification of SVM and Decision Tree classifiers on all 10 digits
```
# Train SVM and Decision Tree classifiers
svm_classifier = train_svm_classifier(X_train, Y_train)
dt_classifier = train_decision_tree_classifier(X_train, Y_train)

# Testing SVM and Decision Tree classifiers performance on test data set
svm_accuracy = test_svm_classifier(svm_classifier, X_test, Y_test)
dt_accuracy = test_decision_tree_classifier(dt_classifier, X_test, Y_test)
print(f"SVM classifier accuracy on test data set: {svm_accuracy * 100:.2f}%")
print(f"Decision Tree classifier accuracy on test data set: {dt_accuracy * 100:.2f}%")

# Testing SVM and Decision Tree classifiers performance on train data set
svm_accuracy = test_svm_classifier(svm_classifier, X_train, Y_train)
dt_accuracy = test_decision_tree_classifier(dt_classifier, X_train, Y_train)
print(f"SVM classifier accuracy on train data set: {svm_accuracy * 100:.2f}%")
print(f"Decision Tree classifier accuracy on train data set: {dt_accuracy * 100:.2f}%")
```

Training and testing SVM and Decision Tree on each unique pair of digits to find the easiest and hardest pairs of digits to separate
```
unique_digits = np.unique(Y)
digit_pairs = list(combinations(unique_digits, 2))

svm_test_accuracies = {}
dt_test_accuracies = {}
svm_train_accuracies = {}
dt_train_accuracies = {}

for pair in digit_pairs:
    # Filter the training and test data for the current pair of digits
    mask_train = np.isin(Y_train, pair)
    X_train_pair = X_train[mask_train]
    Y_train_pair = Y_train[mask_train]
    
    mask_test = np.isin(Y_test, pair)
    X_test_pair = X_test[mask_test]
    Y_test_pair = Y_test[mask_test]

    # Train the SVM and Decision Tree classifiers
    svm_classifier_pair = train_svm_classifier(X_train_pair, Y_train_pair)
    dt_classifier_pair = train_decision_tree_classifier(X_train_pair, Y_train_pair)
    
    # Test the SVM and Decision Tree classiers on test set
    svm_test_accuracy_pair = test_svm_classifier(svm_classifier_pair, X_test_pair, Y_test_pair)
    svm_test_accuracies[pair] = svm_test_accuracy_pair
    
    dt_test_accuracy_pair = test_decision_tree_classifier(dt_classifier_pair, X_test_pair, Y_test_pair)
    dt_test_accuracies[pair] = dt_test_accuracy_pair
    
    # Test the SVM and Decision Tree classiers on train set
    svm_train_accuracy_pair = test_svm_classifier(svm_classifier_pair, X_train_pair, Y_train_pair)
    svm_train_accuracies[pair] = svm_train_accuracy_pair
    dt_train_accuracy_pair = test_decision_tree_classifier(dt_classifier_pair, X_train_pair, Y_train_pair)
    dt_train_accuracies[pair] = dt_train_accuracy_pair
```

Checking accuracy of SVM and Decision Tree on each unique pair of digits
```
# Combine digit pairs with their corresponding test and train accuracies for SVM and Decision Tree classifiers
svm_digit_pair_accuracies = [(pair, svm_test_accuracies[pair], svm_train_accuracies[pair]) for pair in digit_pairs]
dt_digit_pair_accuracies = [(pair, dt_test_accuracies[pair], dt_train_accuracies[pair]) for pair in digit_pairs]

# Sort by highest test accuracy
sorted_svm_digit_pair_accuracies = sorted(svm_digit_pair_accuracies, key=lambda x: x[1], reverse=True)
sorted_dt_digit_pair_accuracies = sorted(dt_digit_pair_accuracies, key=lambda x: x[1], reverse=True)

# Function to print the sorted table
def print_table(sorted_accuracies, classifier_name):
    print(f"{classifier_name} Classifier")
    print("Rank | Digit Pair | Test Set Accuracy | Train Set Accuracy")
    print("-----|------------|------------------|-------------------")
    for rank, (digit_pair, test_accuracy, train_accuracy) in enumerate(sorted_accuracies, start=1):
        print(f"{rank:4} | {digit_pair}    | {test_accuracy * 100:16.2f}% | {train_accuracy * 100:15.2f}%")
    print("\n")

# Print the sorted tables for SVM and Decision Tree classifiers
print_table(sorted_svm_digit_pair_accuracies, "SVM")
print_table(sorted_dt_digit_pair_accuracies, "Decision Tree")
```


---

## Computational Results

### Top 9 SVD Modes

Top 9 SVD Modes
<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemGFirst6SVDModes.png'>
</p>

### Singular Value Spectrum

Singular Value Spectrum
<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemGFirst6SVDModes.png'>
</p>

### Cumulative Variance Explained
<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemGFirst6SVDModes.png'>
</p>

Rank r for 95% variance: 102

MAKRE DUSRE TO JENJFEIFIEFNFE INTERPRETATION OF U E AND V MATRICES

### Projecting digits onto 3 V-modes on a 3D plot

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemGFirst6SVDModes.png'>
</p>

## 2 and 3 Digit LDA classifiers accuracy

| Classifier         | Train Set | Test Set |
|--------------------|-----------|----------|
| LDA ('0', '2')     | 98.57%    | 98.38%   |
| LDA ('1', '3', '5')| 96.11%    | 95.58%   |

## LDA classifier accuracy on each unique pair of digits (sorted by highest accuracy on test set)

| Rank | Digit Pair | Test Set Accuracy | Train Set Accuracy |
|------|------------|-------------------|--------------------|
|   1  | ('6', '7') |           99.75%  |             99.77% |
|   2  | ('1', '4') |           99.65%  |             99.50% |
|   3  | ('0', '1') |           99.64%  |             99.57% |
|   4  | ('0', '7') |           99.58%  |             99.57% |
|   5  | ('6', '9') |           99.57%  |             99.71% |
|   6  | ('1', '6') |           99.55%  |             99.61% |
|   7  | ('0', '4') |           99.52%  |             99.52% |
|   8  | ('1', '9') |           99.48%  |             99.54% |
|   9  | ('5', '7') |           99.45%  |             99.22% |
|  10  | ('0', '9') |           99.18%  |             99.33% |
|  11  | ('1', '7') |           99.18%  |             99.19% |
|  12  | ('3', '4') |           99.03%  |             99.11% |
|  13  | ('3', '6') |           99.02%  |             99.27% |
|  14  | ('1', '5') |           99.00%  |             99.14% |
|  15  | ('0', '3') |           98.94%  |             99.30% |
|  16  | ('0', '6') |           98.89%  |             99.04% |
|  17  | ('4', '8') |           98.85%  |             99.07% |
|  18  | ('4', '6') |           98.83%  |             99.02% |
|  19  | ('4', '5') |           98.75%  |             98.73% |
|  20  | ('0', '8') |           98.67%  |             98.78% |
|  21  | ('1', '3') |           98.63%  |             98.74% |
|  22  | ('6', '8') |           98.58%  |             98.61% |
|  23  | ('4', '7') |           98.47%  |             98.54% |
|  24  | ('5', '9') |           98.45%  |             98.44% |
|  25  | ('7', '8') |           98.44%  |             98.87% |
|  26  | ('3', '7') |           98.42%  |             98.38% |
|  27  | ('0', '2') |           98.38%  |             98.57% |
|  28  | ('2', '9') |           98.34%  |             98.43% |
|  29  | ('0', '5') |           98.31%  |             98.52% |
|  30  | ('2', '7') |           98.20%  |             98.18% |
|  31  | ('2', '4') |           98.18%  |             98.16% |
|  32  | ('1', '2') |           98.09%  |             98.45% |
|  33  | ('5', '6') |           97.78%  |             97.28% |
|  34  | ('8', '9') |           97.62%  |             97.65% |
|  35  | ('3', '9') |           97.59%  |             97.83% |
|  36  | ('2', '5') |           97.43%  |             97.22% |
|  37  | ('2', '6') |           97.42%  |             97.90% |
|  38  | ('2', '3') |           96.96%  |             97.01% |
|  39  | ('2', '8') |           96.75%  |             96.70% |
|  40  | ('1', '8') |           96.49%  |             96.83% |
|  41  | ('3', '8') |           95.92%  |             96.30% |
|  42  | ('7', '9') |           95.76%  |             95.78% |
|  43  | ('3', '5') |           95.45%  |             95.72% |
|  44  | ('5', '8') |           95.34%  |             96.06% |
|  45  | ('4', '9') |           95.18%  |             95.76% |

## SVM and Decision Tree classifiers accuracy on all 10 digits (0-9)

| Classifier          | Test Set Accuracy | Train Set Accuracy |
|---------------------|-------------------|--------------------|
| SVM classifier      | 90.65%            | 91.03%             |
| Decision Tree       | 82.88%            | 100.00%            |

## SVM classifier accuracy on each unique pair of digits (sorted by highest accuracy on test set)

SVM Classifier
Rank | Digit Pair | Test Set Accuracy | Train Set Accuracy
-----|------------|------------------|-------------------
   1 | ('0', '1')    |            99.84% |          100.00%
   2 | ('6', '7')    |            99.84% |          100.00%
   3 | ('1', '6')    |            99.80% |          100.00%
   4 | ('6', '9')    |            99.71% |          100.00%
   5 | ('1', '4')    |            99.65% |          100.00%
   6 | ('1', '7')    |            99.46% |           99.94%
   7 | ('3', '6')    |            99.41% |          100.00%
   8 | ('0', '7')    |            99.40% |          100.00%
   9 | ('0', '4')    |            99.32% |          100.00%
  10 | ('0', '3')    |            99.32% |           99.73%
  11 | ('1', '9')    |            99.30% |           99.98%
  12 | ('1', '5')    |            99.30% |           99.85%
  13 | ('5', '7')    |            99.28% |           99.54%
  14 | ('3', '4')    |            99.05% |           99.73%
  15 | ('4', '5')    |            99.01% |           99.20%
  16 | ('0', '6')    |            99.01% |           99.44%
  17 | ('4', '6')    |            98.98% |           99.41%
  18 | ('4', '7')    |            98.97% |           99.10%
  19 | ('6', '8')    |            98.97% |           99.42%
  20 | ('1', '2')    |            98.95% |           99.45%
  21 | ('0', '8')    |            98.91% |           99.46%
  22 | ('4', '8')    |            98.88% |           99.58%
  23 | ('1', '3')    |            98.88% |           99.64%
  24 | ('7', '8')    |            98.82% |           99.49%
  25 | ('3', '7')    |            98.80% |           99.06%
  26 | ('5', '9')    |            98.75% |           98.86%
  27 | ('2', '9')    |            98.67% |           99.26%
  28 | ('0', '2')    |            98.66% |           99.33%
  29 | ('0', '9')    |            98.65% |          100.00%
  30 | ('2', '7')    |            98.56% |           98.89%
  31 | ('1', '8')    |            98.44% |           98.87%
  32 | ('2', '4')    |            98.43% |           98.77%
  33 | ('8', '9')    |            98.33% |           98.58%
  34 | ('3', '9')    |            98.27% |           98.91%
  35 | ('0', '5')    |            98.21% |           98.74%
  36 | ('5', '6')    |            98.00% |           98.30%
  37 | ('2', '6')    |            97.87% |           98.68%
  38 | ('2', '5')    |            97.85% |           97.90%
  39 | ('2', '3')    |            97.56% |           97.40%
  40 | ('2', '8')    |            97.26% |           97.70%
  41 | ('4', '9')    |            96.70% |           97.00%
  42 | ('3', '8')    |            96.33% |           97.10%
  43 | ('7', '9')    |            96.24% |           96.40%
  44 | ('3', '5')    |            95.97% |           96.58%
  45 | ('5', '8')    |            95.88% |           96.59%

## Decision Tree classifier accuracy on each unique pair of digits (sorted by highest accuracy on test set)

Decision Tree Classifier
Rank | Digit Pair | Test Set Accuracy | Train Set Accuracy
-----|------------|------------------|-------------------
   1 | ('0', '1')    |            99.53% |          100.00%
   2 | ('1', '4')    |            98.98% |          100.00%
   3 | ('6', '7')    |            98.96% |          100.00%
   4 | ('1', '5')    |            98.69% |          100.00%
   5 | ('1', '7')    |            98.66% |          100.00%
   6 | ('1', '9')    |            98.65% |          100.00%
   7 | ('0', '4')    |            98.60% |          100.00%
   8 | ('0', '7')    |            98.58% |          100.00%
   9 | ('1', '3')    |            98.50% |          100.00%
  10 | ('1', '6')    |            98.36% |          100.00%
  11 | ('0', '9')    |            98.12% |          100.00%
  12 | ('1', '2')    |            97.87% |          100.00%
  13 | ('3', '6')    |            97.82% |          100.00%
  14 | ('3', '4')    |            97.81% |          100.00%
  15 | ('0', '3')    |            97.66% |          100.00%
  16 | ('6', '9')    |            97.63% |          100.00%
  17 | ('1', '8')    |            97.61% |          100.00%
  18 | ('5', '7')    |            97.50% |          100.00%
  19 | ('0', '8')    |            97.31% |          100.00%
  20 | ('4', '6')    |            97.12% |          100.00%
  21 | ('3', '7')    |            97.11% |          100.00%
  22 | ('2', '7')    |            96.90% |          100.00%
  23 | ('0', '2')    |            96.80% |          100.00%
  24 | ('2', '4')    |            96.68% |          100.00%
  25 | ('0', '6')    |            96.67% |          100.00%
  26 | ('2', '9')    |            96.50% |          100.00%
  27 | ('6', '8')    |            96.42% |          100.00%
  28 | ('2', '6')    |            96.38% |          100.00%
  29 | ('7', '8')    |            96.34% |          100.00%
  30 | ('4', '5')    |            96.29% |          100.00%
  31 | ('0', '5')    |            96.25% |          100.00%
  32 | ('5', '6')    |            96.20% |          100.00%
  33 | ('2', '5')    |            95.65% |          100.00%
  34 | ('3', '9')    |            95.57% |          100.00%
  35 | ('4', '8')    |            95.49% |          100.00%
  36 | ('4', '7')    |            95.43% |          100.00%
  37 | ('5', '9')    |            95.13% |          100.00%
  38 | ('8', '9')    |            94.59% |          100.00%
  39 | ('2', '3')    |            94.20% |          100.00%
  40 | ('2', '8')    |            93.82% |          100.00%
  41 | ('3', '5')    |            92.47% |          100.00%
  42 | ('5', '8')    |            92.24% |          100.00%
  43 | ('3', '8')    |            91.15% |          100.00%
  44 | ('7', '9')    |            91.06% |          100.00%
  45 | ('4', '9')    |            89.07% |          100.00%








---

## Summary and Conclusions

