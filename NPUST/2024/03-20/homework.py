def is_banned_word(word: str):
    banned_words = ["花花", "雞蛋", "小熊維尼"]
    return word in banned_words


def category_friends():
    colors = {
        "黃色": "是會永遠記得你的人",
        "白色": "是真心對待你的朋友",
        "橘色": "是你愛的人",
        "綠色": "是你的靈魂伴侶",
        "紅色": "是你不會忘的人"
    }

    print("請在以下的顏色中填入一位朋友名字：")
    names = {color: input(f"{color}：") for color in colors}
    print("\n\n以下是測驗結果\n")

    for color, comment in colors.items():
        print(f"{names[color]} {comment}")


def exam_interaction():
    questions = [
        {
            "question": """
下列何者的資料型別為浮點數？
（Ａ）20
（Ｂ）10.0
（Ｃ）” 20.3”
（Ｄ）’30’
        """,
            "answer": "B"
        },
        {
            "question": """
以下哪段程式會發生錯誤？
（Ａ）20 ＋ ’10’
（Ｂ）10 ＋ int（12）
（Ｃ）str（10）＋ str（’26’）
（Ｄ）True＋int（”21”）
        """,
            "answer": "A"
        },
        {
            "question": """
In[1]: X = 6
In[2]: Y = ( )
In[3]: X - Y
Out[3]: 4
            """,
            "answer": 2
        },
        {
            "question": """
In[1]: X = 7
In[2]: X( )=5
In[3]: print(X)
Out[3]: 12
            """,
            "answer": "+"
        }
    ]
    question_name = ("選擇題一", "選擇題二", "填充題一", "填充題二")
    chosen_question = input(f"請從以下題目中選擇一題{question_name}\n請輸入文字：")

    if chosen_question in question_name:
        question_index = question_name.index(chosen_question)
        current_question = questions[question_index]
        print(current_question["question"].strip())
        answered = input("請輸入答案：")
        print(f"你答對了嗎？{answered == current_question['answer']}")


if __name__ == '__main__':
    # first homework
    is_input_illegal = is_banned_word(input("請輸入一句話："))
    print("你說的話含有禁語：{}".format(is_input_illegal))

    # second homework
    category_friends()

    # third homework
    exam_interaction()
