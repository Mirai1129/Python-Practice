a = []

for i in range(10):
    if i % 2 == 0:
        a.append(i)

print(a)

a = [i for i in range(10) if i % 2 == 0]
print(a)
