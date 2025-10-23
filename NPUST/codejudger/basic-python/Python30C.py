text = input()
target = input()
result = 0

for char in text:
    if char in target:
        result += 1

print(result)
