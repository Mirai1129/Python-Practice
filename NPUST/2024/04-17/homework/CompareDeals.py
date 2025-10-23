def find_min_price(*args):
    return min(args)


def main():
    deals = find_min_price(1, 23, 58, 96, 10, 60)
    print(deals)


if __name__ == '__main__':
    main()
