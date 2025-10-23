def transfer_coins() -> None:
    hundred = int(input("How many hundred coins do you have? "))
    five_hundred = int(input("How many five hundred do you have? "))
    thousand = int(input("How many thousand do you have? "))
    money: int = hundred * 100 + five_hundred * 500 + thousand * 1000

    if money % 100 != 0:
        print("只接受紙鈔 (100, 500, 1000)")
        return

    coin_type_1: int = int(input("請選擇硬幣面額個數 (1): "))
    coin_type_5: int = int(input("請選擇硬幣面額個數 (5): "))
    coin_type_10: int = int(input("請選擇硬幣面額個數 (10): "))
    coin_type_50: int = int(input("請選擇硬幣面額個數 (50): "))
    fifty: int = 0
    ten: int = 0
    five: int = 0
    one: int = 0

    transfer_coin: int = coin_type_1 + coin_type_5 * 5 + coin_type_10 * 10 + coin_type_50 * 50
    if transfer_coin <= money:
        fifty += coin_type_50
        money -= fifty * 50
        ten += coin_type_10
        money -= ten * 10
        five += coin_type_5
        money -= five * 5
        one += coin_type_1
        money -= one * 1

    else:
        print("你的錢不夠你換")
        return

    if money >= 50:
        fifty += money // 50
        money -= (fifty - coin_type_50) * 50
    if money >= 10:
        ten += money // 10
        money -= (ten - coin_type_10) * 10
    if money >= 5:
        five += money // 5
        money -= (five - coin_type_5) * 5
    if money >= 1:
        one += money
        money -= (one - coin_type_1)

    print(f"50: {fifty}\n10: {ten}\n5: {five}\n1: {one}")


def main() -> None:
    transfer_coins()


if __name__ == '__main__':
    main()
