from nltk import RegexpTokenizer, download, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame

LEMMATIZER = None
STOP_WORDS = set()


def init_nltk() -> None:
    global LEMMATIZER
    global STOP_WORDS

    if not all((LEMMATIZER, STOP_WORDS)):
        for i in ("punkt_tab", "wordnet", "stopwords"):
            download(i, download_dir=".venv/nltk_data")

        LEMMATIZER = WordNetLemmatizer()
        STOP_WORDS = set(stopwords.words("russian"))
        STOP_WORDS.update(
            (
                "маоу",
                "сош",
                "окпд",
                "для",
                "нужд",
                "гапоу",
                "рх",
                "спт",
            )
        )


def regex_tokenizer(sent: str):
    return RegexpTokenizer(r"\w+").tokenize(sent)


def clean_text(text: str):
    words = regex_tokenizer(sent=text.lower().strip())
    words = word_tokenize(" ".join(words), language="russian")
    words = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
    return " ".join(words)


def preprocess_dataframe(df: DataFrame) -> DataFrame:
    init_nltk()
    df["clean_text"] = df["text"].apply(clean_text)
    return df
