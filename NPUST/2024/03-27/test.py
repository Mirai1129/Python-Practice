# 小宇有記帳的習慣，他會把每一餐的花費記錄下來，
# 請幫他寫一個程式，能夠分別輸入一天三餐的花費，並將費用從小到大排序好在顯示出來。

costs = sorted(map(int, input("Enter the cost of three days (separated by a comma): ").split(',')))
# for i, cost in enumerate(costs):
#     print(f"Day{i + 1}: {cost}")
[print(f"Day{i+1}: {cost}") for i, cost in enumerate(costs)]
