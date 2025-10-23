def print_id_number_first_3_digits(id_number):
    print(id_number[0:3])


def check_repeat_alphabets():
    word1 = input("Enter the word 1:")
    word2 = input("Enter the word 2:")
    print(f"重複的字尾 {set(word1) & set(word2)}")


def upper_word():
    word = input("Enter a word:")
    print(word[::-1])


if __name__ == '__main__':
    print_id_number_first_3_digits(input())
    check_repeat_alphabets()
    upper_word()
