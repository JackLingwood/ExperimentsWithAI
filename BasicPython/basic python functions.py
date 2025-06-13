import math
import matplotlib.pyplot as plt
import random

# Version 1..0.0
# This script demonstrates basic Python functionalities including mathematical operations,


print("Hello, World!")  # Output: Hello, World!
def factorial(n):   
    """Calculates the factorial of a non-negative integer n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print("Factorial of 5 is:", factorial(5))  # Output: 120

def fibonacci(n):
    ###Returns the nth Fibonacci number.###
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print("Fibonacci number at position 10 is:", fibonacci(10))  # Output: 55

def solve_quadratic(a, b, c):
    """Solves ax^2 + bx + c = 0 and returns the real roots as a tuple."""
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        return (root1, root2)
    elif discriminant == 0:
        root = -b / (2*a)
        return (root,)
    else:
        return ()  # No real roots
    

roots = solve_quadratic(1, -3, 2)

print("For the equation x^2 - 3x + 2 = 0, the roots are:")
print(roots)  # Output: (2.0, 1.0)


def fourier_series(f, n_terms, x, period=2*math.pi):
    """
    Approximates a periodic function f(x) using its Fourier series expansion.

    Parameters:
        f (function): The function to approximate.
        n_terms (int): Number of Fourier terms to use.
        x (float): The point at which to evaluate the approximation.
        period (float): The period of the function (default: 2*pi).

    Returns:
        float: The Fourier series approximation at point x.
    """
    a0 = (2 / period) * integrate(f, 0, period)
    result = a0 / 2
    for n in range(1, n_terms + 1):
        an = (2 / period) * integrate(lambda t: f(t) * math.cos(2 * math.pi * n * t / period), 0, period)
        bn = (2 / period) * integrate(lambda t: f(t) * math.sin(2 * math.pi * n * t / period), 0, period)
        result += an * math.cos(2 * math.pi * n * x / period) + bn * math.sin(2 * math.pi * n * x / period)
    return result

def integrate(func, a, b, steps=1000):
    """Numerically integrates func from a to b using the trapezoidal rule."""
    h = (b - a) / steps
    total = 0.5 * (func(a) + func(b))
    for i in range(1, steps):
        total += func(a + i * h)
    return total * h


# Example: Approximate f(x) = x on [0, 2*pi] using 10 Fourier terms at x = 1.0

f = lambda x: x
approx = fourier_series(f, n_terms=10, x=1.0)
print(approx)



def plot_random_numbers():
    # Generates and plots 100 random numbers on a graph.
    data = [random.random() for _ in range(100)]
    plt.figure(figsize=(10, 4))
    plt.plot(data, marker='o', linestyle='-', color='r')
    plt.title("Plot of 100 Random Numbers")
    plt.xlabel("Index")
    plt.ylabel("Random Value")
    plt.grid(True)    
    plt.show()


plot_random_numbers()