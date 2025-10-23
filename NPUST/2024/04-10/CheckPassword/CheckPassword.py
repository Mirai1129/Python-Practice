from flask_bcrypt import Bcrypt
import os


# hashing password to prevent previous danger
# Users should not be told where their password is wrong

class CheckPassword:
    def __init__(self, hashed_password):
        self.hashed_password = hashed_password

    def check(self, password):
        bcrypt = Bcrypt()
        if bcrypt.check_password_hash(self.hashed_password, password):
            print("Correct password")
        else:
            print("Incorrect password")


def main():
    bcrypt = Bcrypt()
    correct_password = os.getenv("PASSWORD")
    hashed_password = bcrypt.generate_password_hash(password=correct_password).decode('utf-8')
    checker = CheckPassword(hashed_password)
    password_input = input("Enter your password: ")
    checker.check(password_input)


if __name__ == "__main__":
    main()
