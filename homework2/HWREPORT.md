# Exploring Correlation and Dimensionality Reduction Techniques on Yalefaces Dataset

**Author**:

Ara Vardanyan

**Abstract**:

This report investigates relationships between 2414 grayscale face images using techniques such as correlation matrices, eigenvectors, and Singular Value Decomposition (SVD). We analyze image correlations, identify highly correlated and uncorrelated pairs, and compute principal component directions. Comparisons between eigenvectors and SVD modes are conducted, followed by an examination of variance captured by SVD modes. The study provides insights into image relationships under different lighting conditions, demonstrating the effectiveness of these techniques in image analysis.

---

## Introduction

In this report, we investigate the relationships between face images using a dataset containing 39 different faces with 65 lighting scenes per face, resulting in a total of 2414 grayscale images downsampled to 32x32 pixels. We explore various techniques to analyze and visualize the dataset, including calculating correlation matrices, computing eigenvectors, and applying Singular Value Decomposition (SVD).

First, we compute a 100x100 correlation matrix for the first 100 images in the dataset and visualize the results using pcolor. We identify the most highly correlated and uncorrelated image pairs and visualize them. Next, we compute a 10x10 correlation matrix for a specified set of images and visualize the resulting matrix.

Further, we create the matrix `Y = XX^T` and find the first six eigenvectors with the largest magnitude eigenvalue. We perform SVD on matrix X to obtain the first six principal component directions. A comparison between the first eigenvector and the first SVD mode is conducted by calculating the norm of the difference between their absolute values.

Lastly, we compute the percentage of variance captured by each of the first six SVD modes and visualize the corresponding modes. This study provides insights into the relationships between face images under different lighting conditions and demonstrates the effectiveness of various techniques in analyzing and processing image data.

---

## Theoretical Background

In this section, we provide a brief theoretical background on the core concepts used in this study, namely correlation, eigenvectors, and Singular Value Decomposition (SVD).

### Correlation

Correlation is a statistical measure that quantifies the degree to which two variables are related. It indicates both the strength and direction of the relationship between the variables. In the context of image analysis, correlation can be used to assess the similarity between two images by comparing their pixel intensities. A greater correlation coefficient signifies a stronger positive relationship.

### Eigenvectors and Eigenvalues

Eigenvectors and eigenvalues are fundamental concepts in linear algebra and have important applications in various fields, including machine learning and data analysis. Given a square matrix A, a non-zero vector v is an eigenvector of A if it satisfies the following equation:

`Av = λv`

Here, λ is a scalar value known as the eigenvalue corresponding to the eigenvector v. In the context of image analysis, eigenvectors and eigenvalues can be employed to perform dimensionality reduction and identify the principal components that capture the most variance in the data.

### Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is a powerful linear algebra technique that decomposes a matrix into the product of three matrices: U, S, and `V^T`. Given a matrix A, the SVD can be represented as:

`A = UΣV^T`

where U and V are orthogonal matrices, and Σ is a diagonal matrix containing singular values in descending order. SVD is widely used in various applications, including image processing, data compression, and dimensionality reduction. In the context of this study, SVD is employed to identify principal component directions that capture the most variance in the dataset.

By understanding these key concepts, we can effectively apply correlation, eigenvectors, and Singular Value Decomposition to analyze the relationships between the images in the Yalefaces dataset and perform dimensionality reduction.

---


## Algorithm Implementation and Development

Initialization of Yalefaces Dataset
```
results = loadmat('./yalefaces.mat')
X = results['X']
```


### Problem (a): Computing and plotting the correlation matrix between the first 100 faces

Extracting the first 100 columns and computing the correlation matrix
```
X_100 = X[:, :100]

C = np.dot(X_100.T, X_100)
```

Plotting the correlation matrix using pcolor
```
plt.pcolor(C)
color_bar = plt.colorbar()
color_bar.set_label('Correlation Coefficient')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.title('Correlation Matrix (100x100)')
plt.show()
```


### Problem (b): Plotting most and least correlated faces

Defining a function to plot the image of a face
```
def plot_image(index, title):
    img = X[:, index].reshape(32, 32)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
```

Specifying the most and least correlated faces
```
highest_corr = [86, 88]
lowest_corr = [19, 96]
```

Plotting the pairs of images
```
plt.figure(figsize=(10, 5))
plt.suptitle('Highest Correlation Images')

plt.subplot(1, 2, 1)
plot_image(highest_corr[0], f'Highest Correlation Image {highest_corr[0] + 1}')
plt.subplot(1, 2, 2)
plot_image(highest_corr[1], f'Highest Correlation Image {highest_corr[1] + 1}')

plt.figure(figsize=(10, 5))
plt.suptitle('Lowest Correlation Images')

plt.subplot(1, 2, 1)
plot_image(lowest_corr[0], f'Lowest Correlation Image {lowest_corr[0] + 1}')
plt.subplot(1, 2, 2)
plot_image(lowest_corr[1], f'Lowest Correlation Image {lowest_corr[1] + 1}')

plt.show()
```


### Problem (c): Computing and plotting the correlation matrix for specified images

Specifying the images
```
img_indices = [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]
```

Extracting the specified images from dataset and computing the 10x10 correlation matrix
```
X_selected = X[:, indices]

C = np.dot(X_selected.T, X_selected)
```

Plotting the correlation matrix using pcolor
```
plt.pcolor(C)
color_bar = plt.colorbar()
color_bar.set_label('Correlation Coefficient')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.title('Correlation Matrix (10x10)')
plt.show()
```

### Problem (d): Creating the matrix `Y = XX^T` and finding the first 6 eigenvectors with the largest magnitude eigenvalues


Computing the matrix `Y = XX^T`
```
Y = np.dot(X, X.T)
```

Computing the eignevalues and eigenvectors of Y and gathering the first 6 eigenvectors with the largest magnitude eigenvalues
```
# Compute eigenvalues and eigenvectors of Y
eigenvalues, eigenvectors = np.linalg.eigh(Y)

# Sort eigenvalues and eigenvectors by descending order of eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# Get the first six eigenvectors
first_six_eigenvectors = eigenvectors[:, :6]
```

### Problem (e): Singular value decomposition of X and finding the first 6 principal component directions


```
# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Select the first six principal component directions
first_six_principal_components = Vt[:6, :]
```

### Problem (f): Computing the norm of the difference of absolute values of the first eigenvector `v_1` and the first SVD mode `u_1`

Gathering the first eigenvector `v_1` from the first six eigenvectors of `Y = XX^T` found earlier
```
v1 = first_six_eigenvectors[:, 0]
```

Gathering the first SVD mode `u_1`
```
u1 = U[:, 0]
```

Computing the norm of the difference of the absolute values of `v_1` and `u_1`
```
norm_diff = np.linalg.norm(np.abs(v1) - np.abs(u1))
```


### Problem (g): Compute the percentage of variance captured by each of the first 6 SVD modes and plotting the first 6 SVD modes

Computing the percentage of variance captured by each of the first 6 SVD modes
```
variance_ratios = (S[:6] ** 2) / np.sum(S ** 2) * 100
```

Plotting the first 6 SVD modes
```
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat, start=1):
    mode_image = U[:, i - 1].reshape(32, 32)
    ax.imshow(mode_image, cmap='gray')
    ax.set_title(f"SVD Mode {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## Computational Results

### Problem (a): Computing and plotting the correlation matrix between the first 100 faces

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemACorrelationMatrix.png'>
</p>

### Problem (b): Plotting most and least correlated faces

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemBMostAndLeastCorrelatedFaces.png'>
</p>

### Problem (c): Computing and plotting the correlation matrix for specified images

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemCCorrelationMatrixSpecifiedImages.png'>
</p>

### Problem (f): Computing the norm of the difference of absolute values of the first eigenvector `v_1` and the first SVD mode `u_1`

    Norm of the difference of absolute values of v1 and u1:
    1.1988821911391694e-15

### Problem (g): Compute the percentage of variance captured by each of the first 6 SVD modes and plotting the first 6 SVD modes

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/a765c023cce8d71a7b5561b259970bd9a6b45e92/homework2/figures/ProblemGFirst6SVDModes.png'>
</p>

| SVD mode | Percentage of variance captured |
|---------:|:--------------------------------|
|        1 | 72.93%                          |
|        2 | 15.28%                          |
|        3 | 2.57%                           |
|        4 | 1.88%                           |
|        5 | 0.64%                           |
|        6 | 0.59%                           |

---

## Summary and Conclusions

In this report, we investigated the relationships between face images in the Yalefaces dataset using techniques such as correlation matrices, eigenvectors, and Singular Value Decomposition (SVD). Our analysis demonstrated that these techniques can provide valuable insights into the relationships between images under different lighting conditions.

We computed a 100x100 correlation matrix for the first 100 images in the dataset and identified the most highly correlated and uncorrelated image pairs. The most correlated pair had similar lighting and face orientation, while the least correlated pair had significantly different lighting conditions. We also computed a 10x10 correlation matrix for a specified set of images and visualized the resulting matrix.

In our exploration of dimensionality reduction techniques, we created the matrix `Y = XX^T` and found the first six eigenvectors with the largest magnitude eigenvalue. We then performed SVD on matrix X to obtain the first six principal component directions. The comparison between the first eigenvector and the first SVD mode showed a very small difference in the norm of their absolute values, indicating that these two methods produce similar results.

Lastly, we computed the percentage of variance captured by each of the first six SVD modes, finding that the first mode captured 72.93% of the variance, and the second mode captured 15.28%. The other modes captured a smaller percentage of the variance. The visualization of the first six SVD modes provided insight into the principal components of the dataset. The variance captured by the first six SVD modes demonstrate how SVD enables us to capture the feature space accurately while reducing the dimensionality of the data resulting in cheaper computation.

In conclusion, our analysis demonstrated the effectiveness of correlation, eigenvectors, and Singular Value Decomposition in understanding the relationships between images in the Yalefaces dataset and performing dimensionality reduction. These techniques can be applied to various image analysis and processing tasks, providing valuable insights into image relationships and enabling more efficient data representation.
