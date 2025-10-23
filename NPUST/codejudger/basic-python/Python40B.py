numbers = []

for i in range(6):
    number = int(input(""))
    numbers.append(number)

a, b, c, d, e, f = numbers

if a >= d and b >= e and c >= f:
    print("Yes")
else:
    print("No")
