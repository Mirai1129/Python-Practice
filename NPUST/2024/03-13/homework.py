def isAccountCorrect():
    accounts = {
        "admin": "1234",
        "dog": "000"
    }

    account = input("account:")
    password = input("password:")
    success = accounts.get(account) == password
    print("login success?", success)


def oxygen():
    length = int(input('length:'))

    oxygenRate = (length - 504.9) / 44.73

    print(f"您的吸氧量為：{str(oxygenRate)}")


def countAge():
    # 輸入出生年月日和現在年月日，計算 user 現在幾歲
    birthday = input("輸入出生年月日(用 - 隔開)：")
    birthYear, birthMonth, birthDate = map(int, birthday.split("-"))

    nowDay = input("輸入現在年月日(用 - 隔開)：")
    nowYear, nowMonth, nowDate = map(int, nowDay.split("-"))

    age = nowYear - birthYear - ((nowMonth, nowDate) < (birthMonth, birthDate))
    print("你現在", age, "歲。")


if __name__ == '__main__':
    isAccountCorrect()
    oxygen()
    countAge()
