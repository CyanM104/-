import numpy as np

x = float(input("Enter the value of x: "))
i = 0
result = 0.0
while 2 * i + 1 < 10:
    result += (-1) ** i / np.factorial(2 * i + 1) * x ** (2 * i + 1)
    i += 1

print("Result of the series is:", result)
