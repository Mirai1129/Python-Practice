class ArticleEncryptor:
    def __init__(self, input_file: str, student_id: str, shift: int = 1):
        self.input_file = input_file
        self.student_id = student_id
        self.shift = shift

    def _shift_text(self, text: str) -> str:
        encrypted_text = ""
        for char in text:
            if char.isalpha():
                shifted = chr(((ord(char) - ord('a' if char.islower() else 'A') + self.shift) % 26) + ord('a' if char.islower() else 'A'))
                encrypted_text += shifted
            else:
                encrypted_text += char
        return encrypted_text

    def encrypt_article(self) -> str:
        with open(self.input_file, "r") as file:
            original_text = file.read()
        encrypted_text = self._shift_text(original_text)
        return encrypted_text

    def write_to_file(self, encrypted_text: str) -> None:
        output_file = self.student_id + ".txt"
        with open(output_file, "w") as file:
            file.write(encrypted_text)


def main() -> None:
    input_file = "The Frog Prince.txt"
    student_id = "B11156055"
    article_encryptor = ArticleEncryptor(input_file, student_id)
    encrypted_text = article_encryptor.encrypt_article()
    article_encryptor.write_to_file(encrypted_text)


if __name__ == "__main__":
    main()
