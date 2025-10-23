import math

a = int(input())
b = int(input())

result = math.sqrt(math.pow(a, 2) + (9 * a * b) + math.pow(b, 2))
print(f"The result is {result:.3f}")
