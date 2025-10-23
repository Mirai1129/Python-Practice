student_score = [int(input(f"請輸入第 {i + 1} 同學的成績：")) for i in range(5)]
total_score = sum(student_score)
average_score = total_score / 5

print(f"本班總成績：{total_score} 分，平均成績：{average_score:.2f} 分")

# student_score = []
#
# for i in range(5):
#     student_score.append(int(input("請輸入第" + str(i + 1) + "同學的成績：")))
#
# total_score = sum(student_score)
# average_score = total_score / 5
#
# print(f"本班總成績：{total_score} 分，平均成績：{average_score:.2f} 分")
