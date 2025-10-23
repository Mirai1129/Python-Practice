numbers = 0

for i in range(6):
    number = int(input(""))
    numbers += number

if numbers % 2 == 0 and numbers % 3 == 0:
    print("Yes")
else:
    print("No")
