'''Problem: A bank processes one million customer transactions. The system needs to convert each transaction amount from Kenyan Shillings (KES) to US Dollars (USD) and calculate the square of each converted amount for a financial analysis.

Solution approach: Use NumPy vectorized expressions:
- amounts_usd = amounts_kes / exchange_rate converts all transaction amounts from KES to USD.
- squared = amounts_usd ** 2 calculates the square of each converted transaction amount.'''


import numpy as np
import timeit

# Simulated bank transactions in Kenyan Shillings
rng = np.random.default_rng(42)
amounts_kes = rng.uniform(100, 100000, 1_000_000)

# Example exchange rate (KES per USD)
exchange_rate = 130.0

def loop_version():
    squared = []
    for value in amounts_kes:
        amount_usd = value / exchange_rate
        squared.append(amount_usd ** 2)
    return squared

def vectorized_version():
    amounts_usd = amounts_kes / exchange_rate
    return amounts_usd ** 2

# Measure execution time
loop_time = timeit.timeit(loop_version, number=1)
vector_time = timeit.timeit(vectorized_version, number=3) / 3

# Generate results
result = vectorized_version()

print("First five squared transaction values:", result[:5])
print(f"Loop time: {loop_time:.4f} s")
print(f"Vectorized time: {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time/vector_time:.1f}x")
