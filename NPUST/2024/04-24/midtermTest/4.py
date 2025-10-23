def phone_ship(time: float | int) -> float:
    ship = time * 0.5
    if time <= 0:
        print("時間應該要 >0")
        return 0

    if time < 800:
        return ship
    elif 800 <= time <= 1500:
        return ship * 0.95
    elif time > 1500:
        return ship * 0.8


def main() -> None:
    ship = phone_ship(float(input("請輸入通話分鐘: ")))
    if ship == 0:
        return
    print(ship)


if __name__ == '__main__':
    main()
