# 試撰寫一程式完成下列要求
# 題目：超商集點
# 說明： 輸入金額後會告訴你要給幾點。如果金 額超過 60 元後每 45 元多 1 點，例如： 105 元可獲得 2 點。

def check_points(amount: float):
    point = 0
    if amount >= 60:
        point = amount // 45
    return int(point)


if __name__ == '__main__':
    money = float(input("what is your money? "))
    points = check_points(money)
    print(points)
