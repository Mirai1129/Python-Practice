class NumberConverter:
    CHINESE_NUMBERS: list[str] = ['零', '壹', '貳', '參', '肆', '伍', '陸', '柒', '捌', '玖']
    CHINESE_UNITS: list[str] = ['', '拾', '佰', '仟']
    CHINESE_LARGE_UNITS: list[str] = ['萬']

    @staticmethod
    def arabic_number_to_chinese_number(number: int) -> str:
        if number == 0:
            return '零元整'

        chinese_number = ''
        digit_groups = []

        # 將數字分組，每個組有四個數字
        while number:
            digit_group = number % 10_000
            digit_groups.append(digit_group)
            number //= 10_000

        for i, digit_group in enumerate(digit_groups):
            if i > 0:
                # 只有到萬，只需要判斷 0 1
                chinese_number = NumberConverter.CHINESE_LARGE_UNITS[i - 1] + chinese_number

            # 組合數字
            group_string = ''
            for j in range(len(str(digit_group))):
                digit = digit_group % 10
                digit_group //= 10
                if digit > 0:
                    group_string = NumberConverter.CHINESE_NUMBERS[digit] + NumberConverter.CHINESE_UNITS[
                        j] + group_string
                elif len(group_string) > 0 and group_string[0] != '零':
                    # 如果上一個數字不是零，則在當前數字為零時添加 '零'
                    group_string = '零' + group_string

            chinese_number = group_string + chinese_number

        return chinese_number + '元整'


def main() -> None:
    number: int = int(input("請輸入一組阿拉伯數字: "))
    chinese_number: str = NumberConverter.arabic_number_to_chinese_number(number)
    print(f"中文大寫表示: {chinese_number}")


if __name__ == "__main__":
    main()
