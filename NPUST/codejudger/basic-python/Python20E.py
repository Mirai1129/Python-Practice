aHour = float(input(""))
bHour = float(input(""))

hours = aHour + bHour

if hours < 3:
    print("早上")
elif hours < 9:
    print("下午")
else:
    print("晚上")
