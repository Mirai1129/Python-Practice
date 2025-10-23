def coffee(coffee_type: str):
    coffee_position: dict[str, str] = {
        "藍山": "牙買加",
        "哥倫比亞": "哥倫比亞",
        "曼特寧": "蘇門達臘群島",
        "瓜地馬拉": "安提瓜區"
    }
    if coffee_type in coffee_position:
        print(coffee_position[coffee_type])
    else:
        print("不明")


def how_tall_you_are():
    height: list[float] = [157.5, 160.2, 156.1, 152.5, 156.1, 155, 151, 153.6]
    sorted_height = sorted(height)
    print(sorted_height)
    print(sorted_height[-1] - sorted_height[0])


def if_save_money():
    weekly_payments: list[int] = [1080, 580, 1630, 633, 842, 2174, 301]
    salary: int = 7500
    print(f"一周總共花了: {sum(weekly_payments)}")
    print(f"開銷最多的一天花了: {max(weekly_payments)}")
    if sum(weekly_payments) <= salary * 0.8:
        print("存得到錢")
    else:
        print("這樣存不到錢啦!")


if __name__ == '__main__':
    # homework 1
    searched_coffee: str = input("請問你要查詢哪種咖啡:")
    coffee(searched_coffee)

    # homework 2
    how_tall_you_are()

    # homework 3
    if_save_money()
