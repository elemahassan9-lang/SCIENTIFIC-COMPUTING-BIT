import numpy as np

# Four corners of a square
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

# Change rotation angle from 45 to 30 degrees
angle = np.deg2rad(30)

# Rotation matrix
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])

# Modified scaling values
scale = np.array([
    [1.5, 0.0],
    [0.0, 2.0]
])

# Combine rotation and scaling
transformation = rotation @ scale

# Transform the original points
transformed_points = points @ transformation.T

# Display results
print("Original square coordinates:\n", points)
print("\nRotation matrix:\n", rotation)
print("\nScaling matrix:\n", scale)
print("\nCombined transformation:\n", transformation)
print("\nNew square coordinates:\n", transformed_points)
print("\nDeterminant:", np.linalg.det(transformation))