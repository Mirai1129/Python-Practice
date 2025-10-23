def oddNumber():
    n = int(input("Enter a number: "))
    oddNumberSum = 0
    for i in range(1, n + 1, 2):
        oddNumberSum += i
    print(oddNumberSum)


def multiplicationTable():
    for i in range(2, 10):
        for j in range(1, 10):
            print("{}*{}={:2d}".format(i, j, i*j), end=' ')
        print()
