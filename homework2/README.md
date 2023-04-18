# Exploring Correlation and Dimensionality Reduction Techniques on Yalefaces Dataset

**Author**:  

Ara Vardanyan

**Abstract**:

## Introduction



## Theoretical Background



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

### Problem (d): Creating the matrix `Y=XX^T` and finding the first 6 eigenvectors with the largest magnitude eigenvalues


Computing the matrix `Y=XX^T`
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

Gathering the first eigenvector `v_1` from the first six eigenvectors of `Y=XX^T` found earlier
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


## Computational Results

### Problem (a): Computing and plotting the correlation matrix between the first 100 faces

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework2/figures/ProblemACorrelationMatrix.png'>
</p>

### Problem (b): Plotting most and least correlated faces

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework2/figures/ProblemBMostAndLeastCorrelatedFaces.png'>
</p>

### Problem (c): Computing and plotting the correlation matrix for specified images

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework2/figures/ProblemCCorrelationMatrixSpecifiedImages.png'>
</p>

### Problem (f): Computing the norm of the difference of absolute values of the first eigenvector `v_1` and the first SVD mode `u_1`

    Norm of the difference of absolute values of v1 and u1:
    1.1988821911391694e-15

### Problem (g): Compute the percentage of variance captured by each of the first 6 SVD modes and plotting the first 6 SVD modes

<p>
  <img src='https://github.com/ara-vardanyan/EE-399-Machine-Learning-HW-Reports/blob/fe3d5d7eb7326b19c9d62ad381ceb5f4ce496467/homework2/figures/ProblemGFirst6SVDModes.png'>
</p>

| SVD mode | Percentage of variance captured |
|---------:|:--------------------------------|
|        1 | 72.93%                          |
|        2 | 15.28%                          |
|        3 | 2.57%                           |
|        4 | 1.88%                           |
|        5 | 0.64%                           |
|        6 | 0.59%                           |
 
## Summary and Conclusions

