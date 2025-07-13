import typer

from cl_okpd2.constants import ModelTypes
from cl_okpd2.main import fit_save as ml_fit_save
from cl_okpd2.main import predict as ml_predict

manager = typer.Typer()


@manager.command()
def fit_save(path: str):
    ml_fit_save(ml_model_name=ModelTypes.CAT_BOOST, path=path)


@manager.command()
def predict(text: str):
    ml_predict(ml_model_name=ModelTypes.CAT_BOOST, text=text)


if __name__ == "__main__":
    manager()
