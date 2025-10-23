def lottery_judger(guess_numbers: list[int]):
    lottery_numbers: list[int] = [6, 8, 10, 28, 41, 43, 45]
    special_number: int = lottery_numbers[-1]
    count: int = 0
    is_special_number: bool = False
    for guess_number in guess_numbers:
        if guess_number in lottery_numbers:
            if guess_number == special_number:
                is_special_number = True
            count += 1

    formatted_numbers: str = ', '.join(['{:02d}'.format(number) for number in lottery_numbers])

    print("獎號開出順序：{}，特別號：{}".format(formatted_numbers, special_number))
    print("結果: ", end="")

    if is_special_number:
        if count == 2:
            print("柒獎 2碼+特別號", end="")
        if count == 3:
            print("陸獎 3碼+特別號", end="")
        if count == 4:
            print("肆獎 4碼+特別", end="")
        if count == 5:
            print("貳獎 5碼+特別", end="")
    else:
        if count == 3:
            print("普獎 任三碼", end="")
        if count == 4:
            print("伍獎 4碼", end="")
        if count == 5:
            print("叁獎 5碼", end="")
        if count == 6:
            print("頭獎 6個", end="")
        else:
            print("沒中獎")


def guess_lottery():
    input_numbers: list[int] = []
    try:
        input_numbers: list[int] = [int(input(f"lottery number({number + 1}): ")) for number in range(6)]
    except ValueError:
        print("You must input a number")
    lottery_judger(input_numbers)


def main():
    guess_lottery()


if __name__ == '__main__':
    main()
