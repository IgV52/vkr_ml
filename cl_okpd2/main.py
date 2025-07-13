from os.path import join as os_join

import pandas as pd

from cl_okpd2.constants import CHUNK_SIZE, MAP_OKPD2_CODE, NEW_DATA_DIRNAME, ModelTypes
from cl_okpd2.etl import extract_draft_data, extract_uniq_data
from cl_okpd2.ml import CustomML


def etl(path: str) -> str:
    etl_draft = os_join(NEW_DATA_DIRNAME, "draft_data.csv")
    etl_clear = os_join(NEW_DATA_DIRNAME, "clear_data.csv")
    extract_draft_data(path=path, new_path=etl_draft, chunk_size=CHUNK_SIZE)
    extract_uniq_data(path=etl_draft, new_path=etl_clear)

    return etl_clear


def fit_metrics(ml_model_name: ModelTypes, path_data: str):
    clear_data_path = etl(path=path_data)
    df = pd.read_csv(clear_data_path, names=["text", "code"])
    model = CustomML(df=df, model_type=ml_model_name)
    model.clear_df(
        min_qty=100, grpoup_by_qty=163, farthest_name_classes=["C", "A"]
    )  # 198, 156, 159
    model.fit()
    print()
    print(
        f"Точность модели {ml_model_name} на тесте: {int(round(model.metrics['Accuracy'], 2) * 100)} процентов"
    )
    print(f"Отчет модели {ml_model_name} {model.metrics['Classification_report']}")
    print()
    print(
        f"Точность модели {ml_model_name} при обучении: {int(round(model.metrics['Accuracy_train'], 2) * 100)} процентов"
    )
    print(f"Отчет модели {ml_model_name} {model.metrics['Classification_report_train']}")
    print()


def fit_save(ml_model_name: ModelTypes, path_data: str):
    clear_data_path = etl(path=path_data)
    df = pd.read_csv(clear_data_path, names=["text", "code"])
    model = CustomML(df=df, model_type=ml_model_name)
    model.clear_df(min_qty=100, grpoup_by_qty=163, farthest_name_classes=["C", "A"])
    model.fit()
    model.save_model()


def predict(ml_model_name: ModelTypes, text: str):
    model = CustomML(model_type=ml_model_name, df=pd.DataFrame([text], columns=["text"]))
    model.load_model()
    pred = model.predict()
    print(f"Возможный код ОКПД-2: {pred} - {MAP_OKPD2_CODE.get(pred, f'ОШИБКА {pred}')}")
