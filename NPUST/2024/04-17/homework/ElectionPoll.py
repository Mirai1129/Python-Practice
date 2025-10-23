def election_poll(**kwargs):
    total_votes = sum(kwargs.values())
    for key, value in kwargs.items():
        percentage = (value / total_votes) * 100
        print(f'{key}: {percentage:.2f}%')


def main():
    election_poll(小晴=10, 小雨=18, 小金=7)


if __name__ == '__main__':
    main()
