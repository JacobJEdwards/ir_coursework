from config import VIDEOGAMES_DIR
from parser.reader import read_dir


def main() -> None:
    results = read_dir(VIDEOGAMES_DIR)

    for file_path, result in results.items():
        print(f"File: {file_path}")
        if isinstance(result, str):
            print(f"Error: {result}")
        # else:
        # print(f"Parsed Contents: {result}")


if __name__ == "__main__":
    main()
