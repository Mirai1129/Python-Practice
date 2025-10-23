import math

score = int(input())

# 如果成績是及格的（大於等於 60 分）
if score >= 60:
    result = score + 5
# 如果不及格
else:
    result = math.floor(math.sqrt(score) * 10)  # 使用 math.floor() 對計算結果進行無條件捨去
print(f"{result}")
