import dis


def add(a, b):
    return a + b


def main():
    dis.dis(add)
    dis.dis(lambda x, y: x + y)


if __name__ == '__main__':
    with open(__file__, "rb") as f:
        bytecode = f.read()
    dis.dis(bytecode)
