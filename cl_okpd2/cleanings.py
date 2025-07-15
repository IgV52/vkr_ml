from nltk import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from pandas import DataFrame
from stanza import Pipeline

LEMMATIZER = None
STOP_WORDS = None


def init_nltk() -> None:
    global STOP_WORDS

    if not STOP_WORDS:
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


def init_stanza() -> None:
    global LEMMATIZER

    if not LEMMATIZER:
        LEMMATIZER = Pipeline("ru")


def regex_tokenizer(sent: str):
    return RegexpTokenizer(r"\w+").tokenize(sent)


def lemmatize(text: str) -> str:
    doc = LEMMATIZER(text)
    return " ".join(w.lemma for i in doc.sentences for w in i.words)


def preprocess_lemmatize(df: DataFrame) -> DataFrame:
    init_stanza()
    df["clean_text"] = df["clean_text"].apply(lemmatize)
    return df


def clean_text(text: str):
    words = regex_tokenizer(sent=text.lower().strip())
    words = word_tokenize(" ".join(words), language="russian")
    words = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
    return " ".join(words)


def preprocess_dataframe(df: DataFrame) -> DataFrame:
    init_nltk()
    df["clean_text"] = df["text"].apply(clean_text)
    return df
