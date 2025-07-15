from enum import StrEnum
from os.path import join as os_join

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from matplotlib.patches import Rectangle
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from cl_okpd2.cleanings import preprocess_dataframe, preprocess_lemmatize
from cl_okpd2.constants import MODELS_PATH


class ModelTypes(StrEnum):
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    CAT_BOOST = "cat_boost"


class CustomML:
    def __init__(self, model_type: ModelTypes, df: DataFrame) -> None:
        self.model_type = model_type
        self.vecrotize = TfidfVectorizer()
        self.metrics = {}
        self.model_init()
        self.df = df.copy()

    def model_init(self) -> None:
        match self.model_type:
            case ModelTypes.RANDOM_FOREST:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            case ModelTypes.LOGISTIC_REGRESSION:
                self.model = LogisticRegression()
            case ModelTypes.CAT_BOOST:
                self.model = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.1,
                    depth=3,
                    verbose=0,
                )

    def filtered_fathest_index(self, masks: list[str], df: DataFrame, max_qty: int) -> DataFrame:
        for mask in masks:
            df_mask = df["code"] == mask
            tf_df_draft = df.loc[df_mask].copy()
            df = df.drop(df.loc[df_mask].index).reset_index(drop=True)
            tfidf_matrix = self.vecrotize.fit_transform(tf_df_draft["clean_text"])
            distance = cosine_distances(tfidf_matrix)
            mean_dist = distance.mean(axis=1)
            unique_indices = np.argsort(mean_dist)[::-1][:max_qty]
            df_new = tf_df_draft.iloc[unique_indices]
            df = pd.concat([df, df_new], ignore_index=True)

        return df

    def clear_df(self, min_qty: int, grpoup_by_qty: int, farthest_name_classes: list[str]) -> None:
        df = preprocess_dataframe(df=self.df)
        df = self.filtered_fathest_index(
            masks=farthest_name_classes, df=df, max_qty=int(grpoup_by_qty)
        )
        counts = df["code"].value_counts()
        self.df = df.loc[df["code"].isin(counts.loc[counts >= min_qty].index)].reset_index(
            drop=True
        )
        self.df = preprocess_lemmatize(df=self.df)

    def save_heat_map(self, y_test, predict_x_test, classes, data_type):
        cm = confusion_matrix(y_test, predict_x_test, labels=classes)
        max_value = cm.max()
        background = np.zeros_like(cm)
        np.fill_diagonal(background, 1)
        ax = sns.heatmap(
            cm,
            annot=True,
            cbar=False,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        mask = background == 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if mask[i][j] and cm[i][j]:
                    intensity = cm[i][j] / max_value
                    ax.add_patch(
                        Rectangle((j, i), 1, 1, color=(1, 1 - intensity, 1 - intensity, 0.8))
                    )

        plt.title("Матрица ошибок")
        plt.xlabel("Классы")
        plt.savefig(f"{self.model_type}-{data_type}.png")
        plt.close()

    def fit(self):
        pipline = make_pipeline(self.vecrotize, self.model)
        x_train, x_test, y_train, y_test = train_test_split(
            self.df["clean_text"], self.df["code"], test_size=0.25, random_state=42
        )
        pipline.fit(x_train, y_train)
        predict_x_test = pipline.predict(x_test)
        predict_x_train = pipline.predict(x_train)
        self.metrics["Accuracy"] = pipline.score(x_test, y_test)
        self.metrics["Accuracy_train"] = pipline.score(x_train, y_train)
        self.metrics["Classification_report"] = classification_report(y_test, predict_x_test)
        self.metrics["Classification_report_train"] = classification_report(
            y_train, predict_x_train
        )
        self.save_heat_map(
            y_test=y_test,
            predict_x_test=predict_x_test,
            classes=pipline.classes_,
            data_type="test",
        )
        self.save_heat_map(
            y_test=y_train,
            predict_x_test=predict_x_train,
            classes=pipline.classes_,
            data_type="train",
        )

        return pipline

    def predict(self) -> str:
        df = preprocess_dataframe(df=self.df)
        df = preprocess_lemmatize(df=df)
        res = self.model.predict(self.vecrotize.transform(df["clean_text"]))

        match self.model_type:
            case ModelTypes.CAT_BOOST:
                answer = res[0][0]
            case ModelTypes.LOGISTIC_REGRESSION | ModelTypes.RANDOM_FOREST:
                answer = res[0]

        return answer

    def save_model(self) -> None:
        joblib.dump(os_join(MODELS_PATH, self.vecrotize, f"{self.model_type}.vec"))
        match self.model_type:
            case ModelTypes.CAT_BOOST:
                self.model.save_model(fname=os_join(MODELS_PATH, f"{self.model_type}.cbm"))
            case ModelTypes.LOGISTIC_REGRESSION | ModelTypes.RANDOM_FOREST:
                joblib.dump(self.model, os_join(MODELS_PATH, f"{self.model_type}.pkl"))

    def load_model(self) -> None:
        self.vecrotize = joblib.load(os_join(MODELS_PATH, f"{self.model_type}.vec"))
        match self.model_type:
            case ModelTypes.CAT_BOOST:
                self.model.load_model(fname=os_join(MODELS_PATH, f"{self.model_type}.cbm"))
            case ModelTypes.LOGISTIC_REGRESSION | ModelTypes.RANDOM_FOREST:
                self.model = joblib.load(os_join(MODELS_PATH, f"{self.model_type}.pkl"))
