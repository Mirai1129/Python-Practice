a = 0
b = "Hello World!"

# tuple_a = ('a', 'b')
# list_a = ['a', 'b']
set_a = {
    "a": "apple",
    "b": "banana",
    "c": "cum"
}

print(set_a)
print(set_a.keys())
print(set_a.values())
print(set_a.items())
print(set_a["b"])

for dict_key in set_a.items():
    print(dict_key)

fruit = ["apple", "banana", "cum"]

fruit_index = fruit.index("banana")

print(fruit_index)
