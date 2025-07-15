from nltk import download


def main():
    for i in ("punkt_tab", "wordnet", "stopwords"):
        download(i, download_dir=".venv/nltk_data")


if __name__ == "__main__":
    main()
