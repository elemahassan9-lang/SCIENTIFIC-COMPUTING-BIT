
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

print("ITERATIVE METHODS FOR LINEAR SYSTEMS")
print("1. Original 5-point temperature model")
print("2. Modified 3-point temperature model")
print("3. General 3-variable linear system")

choice = input("Choose a model (1, 2 or 3): ")

if choice == "1":
    A = np.array([
        [ 2., -1.,  0.,  0.,  0.],
        [-1.,  2., -1.,  0.,  0.],
        [ 0., -1.,  2., -1.,  0.],
        [ 0.,  0., -1.,  2., -1.],
        [ 0.,  0.,  0., -1.,  2.]
    ])

    b = np.array([100., 0., 0., 0., 20.])
    model_name = "Original temperature model"

elif choice == "2":
    A = np.array([
        [ 4., -1.,  0.],
        [-1.,  4., -1.],
        [ 0., -1.,  4.]
    ])

    b = np.array([100., 50., 20.])
    model_name = "Modified 3-point model"

elif choice == "3":
    A = np.array([
        [10., -2.,  1.],
        [-1.,  8., -2.],
        [ 2., -1.,  7.]
    ])

    b = np.array([30., 20., 10.])
    model_name = "General linear system"

else:
    print("Invalid selection.")
    raise SystemExit


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

n = len(b)

guess_input = input(
    f"Enter {n} initial guesses separated by commas "
    "(press Enter for zeros): "
)

if guess_input.strip() == "":
    x0 = np.zeros(n)
else:
    x0 = np.array(
        [float(value) for value in guess_input.split(",")]
    )

    if len(x0) != n:
        print("Error: the number of guesses must match the unknowns.")
        raise SystemExit

tol = float(input("Enter tolerance (e.g. 0.000001): "))
max_iter = int(input("Enter maximum iterations (e.g. 500): "))

if tol <= 0 or max_iter <= 0:
    print("Tolerance and iterations must be positive.")
    raise SystemExit

if np.any(np.diag(A) == 0):
    print("Error: diagonal elements cannot be zero.")
    raise SystemExit


# --------------------------------------------------
# JACOBI METHOD
# --------------------------------------------------

def jacobi(A, b, x0, tol=1e-8, max_iter=500):

    D = np.diag(A)
    R = A - np.diagflat(D)

    x = x0.copy()
    errors = []

    for iteration in range(max_iter):

        x_new = (b - R @ x) / D

        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)

        if error < tol:
            return x_new, errors, True

        x = x_new

    return x, errors, False


# --------------------------------------------------
# GAUSS-SEIDEL METHOD
# --------------------------------------------------

def gauss_seidel(A, b, x0, tol=1e-8, max_iter=500):

    x = x0.copy()
    errors = []

    for iteration in range(max_iter):

        old = x.copy()

        for i in range(len(b)):

            x[i] = (
                b[i]
                - A[i, :i] @ x[:i]
                - A[i, i+1:] @ old[i+1:]
            ) / A[i, i]

        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)

        if error < tol:
            return x, errors, True

    return x, errors, False


# --------------------------------------------------
# CALCULATIONS
# --------------------------------------------------

x_j, err_j, converged_j = jacobi(
    A, b, x0, tol, max_iter
)

x_gs, err_gs, converged_gs = gauss_seidel(
    A, b, x0, tol, max_iter
)

x_exact = np.linalg.solve(A, b)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nMODEL:", model_name)

print("\nJacobi solution:")
print(x_j)

print("Jacobi iterations:", len(err_j))
print("Jacobi converged:", converged_j)

print("\nGauss-Seidel solution:")
print(x_gs)

print("Gauss-Seidel iterations:", len(err_gs))
print("Gauss-Seidel converged:", converged_gs)

print("\nDirect solution:")
print(x_exact)


# --------------------------------------------------
# CONVERGENCE GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.semilogy(err_j, label="Jacobi")
plt.semilogy(err_gs, label="Gauss-Seidel")

plt.xlabel("Iteration")
plt.ylabel("Maximum change")
plt.title("Iterative Methods Convergence Comparison")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
