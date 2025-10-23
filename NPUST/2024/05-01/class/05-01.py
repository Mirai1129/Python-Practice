with open("News.txt", "r", encoding="utf-8") as f:
    content = f.read()
    char_count = len(content)

modified_content = content.replace("24", "48")

with open("New2.txt", "w", encoding="utf-8") as f:
    f.write(modified_content)

print("Successfully replaced and saved.")
print("Number of characters in the file:", char_count)
