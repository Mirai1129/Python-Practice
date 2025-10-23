def print_double() -> None:
    # abc = [i**2 for i in range(1, 21) if i % 2 == 0]

    n = 1
    result = []
    double = []
    while n <= 20:
        if n % 2 == 0:
            result.append(n ** 2)
            double.append(n)
            n += 1
        else:
            n += 1
    print(double, sum(result))


def main() -> None:
    print_double()


if __name__ == '__main__':
    main()
