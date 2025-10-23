text = input()

text_length = len(text)

if text_length >= 8:
    status = "OK"
else:
    status = "FAILED"

print(status)
