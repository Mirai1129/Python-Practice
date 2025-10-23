import pandas as pd
from pandas import DataFrame, Series

FILE_PATH = "chicken.csv"
ENCODING = ['Big5']


def read_csv_file() -> DataFrame:
    for encoding in ENCODING:
        try:
            df = pd.read_csv(filepath_or_buffer=FILE_PATH, encoding=encoding)
            return df
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding. Trying another encoding...")


def get_cvs_duplicate(df: DataFrame) -> Series | None | DataFrame:
    return df[df.duplicated(keep=False)]


def main() -> None:
    data = read_csv_file()
    duplicate_df = get_cvs_duplicate(data)
    print(data, duplicate_df)


if __name__ == "__main__":
    main()
