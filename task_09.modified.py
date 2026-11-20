'''Problem:
A university wants to analyse students' academic performance across different subjects, such as Mathematics, Programming, Physics and Statistics. The university wants to identify the main patterns in students' performance and determine which combination of subjects contributes most to overall academic variation.

Solution approach:

Represent students' marks in a data matrix and calculate the covariance matrix of the subjects. Use NumPy to calculate the eigenvalues and eigenvectors of the covariance matrix. The eigenvector corresponding to the largest eigenvalue identifies the principal direction of variation in students' performance, while the eigenvalue indicates how much variation that direction explains'''


import numpy as np

# 1. Students' marks in different subjects
# Columns represent Mathematics, Programming, Physics and Statistics

marks = np.array([
    [80, 85, 75, 82],
    [70, 75, 65, 72],
    [90, 95, 85, 92],
    [60, 65, 55, 62],
    [85, 88, 80, 86],
    [75, 78, 70, 76],
    [95, 98, 90, 96],
    [65, 68, 60, 66]
], dtype=float)

subjects = ["Mathematics", "Programming", "Physics", "Statistics"]

# 2. Calculate the mean mark for each subject
mean_marks = np.mean(marks, axis=0)

# 3. Center the data by subtracting the mean
centered_data = marks - mean_marks

# 4. Calculate the covariance matrix
covariance_matrix = np.cov(centered_data, rowvar=False)

# 5. Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# 6. Arrange eigenvalues in descending order
indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

# 7. Identify the principal component
dominant_eigenvalue = eigenvalues[0]
dominant_eigenvector = eigenvectors[:, 0]

# 8. Calculate the percentage of variation explained
explained_variance = (
    dominant_eigenvalue / np.sum(eigenvalues)
) * 100

# 9. Display the results
print("STUDENT PERFORMANCE ANALYSIS")

print("\nMean Marks:")
for subject, mean in zip(subjects, mean_marks):
    print(f"{subject}: {mean:.2f}")

print("\nCovariance Matrix:")
print(np.round(covariance_matrix, 2))

print("\nEigenvalues:")
print(np.round(eigenvalues, 4))

print("\nDominant Eigenvalue:")
print(round(dominant_eigenvalue, 4))

print("\nDominant Eigenvector:")
for subject, value in zip(subjects, dominant_eigenvector):
    print(f"{subject}: {value:.4f}")

print("\nPercentage of Variation Explained:")
print(f"{explained_variance:.2f}%")

# 10. Verify the eigenvalue equation
verification = np.linalg.norm(
    covariance_matrix @ dominant_eigenvector -
    dominant_eigenvalue * dominant_eigenvector
)

print("\nVerification Error:")
print(f"{verification:.10f}")
