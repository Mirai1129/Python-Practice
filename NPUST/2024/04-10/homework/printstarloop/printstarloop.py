import keyboard
import sys


def try_exit_program():
    if keyboard.is_pressed('ctrl+q') or keyboard.is_pressed('ctrl+c'):
        print("Exiting program...")
        sys.exit()


def print_numbers():
    while True:
        for i in range(1, 21):
            print("*" * i)
            try_exit_program()


def main():
    print_numbers()


if __name__ == '__main__':
    main()
