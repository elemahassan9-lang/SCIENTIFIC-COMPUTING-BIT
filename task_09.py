
'''Problem: A small computer network needs to identify which nodes are structurally most influential from 
their pattern of connections. 
Solution approach: Represent the network as an adjacency matrix, compute its dominant eigenvalue and 
eigenvector, and normalize the dominant eigenvector to obtain an eigenvector-based node-importance 
measure.'''

import numpy as np

# Adjacency matrix for the modified network
# Node order: A, B, C, D
A = np.array([
    [0., 1., 1., 1.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [1., 1., 1., 0.]
])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Find the dominant eigenvalue
index = np.argmax(np.real(eigenvalues))
dominant_value = np.real(eigenvalues[index])

# Obtain the corresponding eigenvector
dominant_vector = np.abs(
    np.real(eigenvectors[:, index])
)

# Normalize the eigenvector
centrality = dominant_vector / dominant_vector.sum()

nodes = ["A", "B", "C", "D"]

print("Dominant eigenvalue:", dominant_value)
print("Eigenvector-based importance:")

for node, score in zip(nodes, centrality):
    print(f"{node}: {score:.4f}")

# Verify Av = lambda*v
v = np.real(eigenvectors[:, index])

print("Verification error:",
      np.linalg.norm(A @ v - dominant_value * v))
