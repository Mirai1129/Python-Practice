import math

DESCRIPTION = """
歡迎使用高級計算機
選擇操作:
1. 加
2. 減
3. 乘
4. 除
5. 平方根
6. 正弦值
7. 餘弦值
8. 正切值
"""


class AdvancedCalculator:
    @staticmethod
    def add(x: float, y: float) -> float:
        """
        計算兩個數字的和。

        :param x: 第一個數字
        :type x: float
        :param y: 第二個數字
        :type y: float

        :return: 兩個數字的和
        :rtype: float
        """
        return x + y

    @staticmethod
    def subtract(x: float, y: float) -> float:
        """
        計算兩個數字的差。

        :param x: 第一個數字
        :type x: float
        :param y: 第二個數字
        :type y: float

        :return: 兩個數字的差
        :rtype: float
        """
        return x - y

    @staticmethod
    def multiply(x: float, y: float) -> float:
        """
        計算兩個數字的積。

        :param x: 第一個數字
        :type x: float
        :param y: 第二個數字
        :type y: float

        :return: 兩個數字的積
        :rtype: float
        """
        return x * y

    @staticmethod
    def divide(x: float, y: float) -> float:
        """
        計算兩個數字的商。

        :param x: 被除數
        :type x: float
        :param y: 除數
        :type y: float

        :return: 兩個數字的商
        :rtype: float

        :raises ValueError: 如果除數為零
        """
        if y == 0:
            raise ValueError("錯誤：除數為零")
        return x / y

    @staticmethod
    def square_root(x: float) -> float:
        """
        計算一個數字的平方根。

        :param x: 要計算平方根的數字
        :type x: float

        :return: 該數字的平方根
        :rtype: float

        :raises ValueError: 如果數字為負數
        """
        if x < 0:
            raise ValueError("錯誤：負數的平方根")
        return math.sqrt(x)

    @staticmethod
    def sine(x: float) -> float:
        """
        計算角度的正弦值。

        :param x: 角度（度）
        :type x: float

        :return: 角度的正弦值
        :rtype: float
        """
        return math.sin(math.radians(x))

    @staticmethod
    def cosine(x: float) -> float:
        """
        計算角度的餘弦值。

        :param x: 角度（度）
        :type x: float

        :return: 角度的餘弦值
        :rtype: float
        """
        return math.cos(math.radians(x))

    @staticmethod
    def tangent(x: float) -> float:
        """
        計算角度的正切值。

        :param x: 角度（度）
        :type x: float

        :return: 角度的正切值
        :rtype: float
        """
        return math.tan(math.radians(x))

    def run_calculator(self, choice: str) -> None:
        """
        根據用戶選擇運行相應的計算操作。

        :param choice: 用戶的選擇（1-8）
        :type choice: str

        :return: None
        """
        operations = {
            '1': lambda: self.add(*map(float, input("輸入兩個數字（以空格分隔）: ").split())),
            '2': lambda: self.subtract(*map(float, input("輸入兩個數字（以空格分隔）: ").split())),
            '3': lambda: self.multiply(*map(float, input("輸入兩個數字（以空格分隔）: ").split())),
            '4': lambda: self.divide(*map(float, input("輸入兩個數字（以空格分隔）: ").split())),
            '5': lambda: self.square_root(float(input("輸入一個數字: "))),
            '6': lambda: self.sine(float(input("輸入角度（度）: "))),
            '7': lambda: self.cosine(float(input("輸入角度（度）: "))),
            '8': lambda: self.tangent(float(input("輸入角度（度）: ")))
        }

        try:
            if choice in operations:
                result = operations[choice]()
                print(f"結果: {result}")
            else:
                print("非法輸入")
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"發生錯誤: {e}")


def main() -> None:
    calculator = AdvancedCalculator()
    print(DESCRIPTION)
    choice = input("輸入你的選擇 (1/2/3/4/5/6/7/8): ")
    calculator.run_calculator(choice)


if __name__ == "__main__":
    main()
