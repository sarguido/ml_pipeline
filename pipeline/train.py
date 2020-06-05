"""
Short description - This module contains code to train the model and save it.

:author: Sarah Guido
"""
import click
import pandas as pd

from typing import Dict, Any, Tuple, Set
from ast import literal_eval
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


def value_check(val: str) -> Any:
    """
    Transform a single string value into proper hyperparameter input.

    :param val: string value from command line input
    :return: properly transformed type
    """
    if val.isdigit():
        assert int(val) >= 0, "Value cannot be transformed into an integer."
        return int(val)

    elif val in ["True", "true", "False", "false"]:
        assert literal_eval(val.title()), "Value cannot be interpreted as a boolean."
        return literal_eval(val.title())

    elif "." in val:
        assert float(val), "Value cannot be transformed into a float."
        return float(val)

    else:
        assert isinstance(val, str), "Value is not a string."
        return val


def format_hyperparameters(params: Tuple[str, ...]) -> Dict:
    """
    Transform command line tuples into dictionary for training model.

    :param params: Command line input for hyperparameters.
    :return: Formatted dictionary of hyperparameters.
    """
    assert isinstance(params, tuple), "Parameters must be in Tuple format."
    assert len(params) > 0, "Cannot pass an empty tuple."

    params_list = [k_v.split("=") for k_v in params]
    param_dict = {}

    for key, value in params_list:
        param_dict[key] = value_check(value)

    return param_dict


def train_and_save(X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   model_filename: str,
                   params: Tuple[str, ...]) -> None:
    """
    Build the Random Forest Regression model. Train on the data
    and serialize the model itself.

    :param X_train: Feature set to train on.
    :param y_train: Values to predict.
    :param model_filename: Path to store model.
    :param params: Hyperparameter dictionary.
    """
    rfr = RandomForestRegressor(verbose=2)

    if params:
        click.echo("Formatting hyperparameters for training.")
        param_dict = format_hyperparameters(params)

        rfr.set_params(**param_dict)
        click.echo(f"Updated the following hyperparameters: {param_dict.keys()}")

    click.echo("Fitting the random forest regression model.")
    rfr.fit(X_train, y_train["price"].values)

    click.echo(f"Saving model to disk.")
    dump(rfr, model_filename)


@click.command()
@click.argument("x_train_path", type=click.Path(exists=True))
@click.argument("y_train_path", type=click.Path(exists=True))
@click.option("--model_filename",
              type=click.Path(),
              default="rfr.joblib",
              help="Filename for saved model. Default is rfr.joblib.")
@click.option("--hyperparameters", "-hp",
              type=str,
              multiple=True,
              help="Optional hyperparameters for random forest regression model. For each parameter you want to enter, "
                   "please pass the -hp flag followed by the hyperparameters in the format parameter=value ex. "
                   "-hp n_estimators=50 -hp criterion=mae. Defaults to None.")
def train_model(x_train_path: str, y_train_path: str, model_filename: str, hyperparameters: Tuple[str, ...] = None):
    """
    Train a Random Forest Regression model and save the model to disk. Please pass in the paths to the training X and Y
    datasets. Model is automatically set to verbose=2 in order to observe training progress. If you'd like to turn this
    behavior off, pass in -hp verbose=0 to the hyperparameter option.
    """
    click.echo("Reading in training data.")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    train_and_save(X_train, y_train, model_filename, hyperparameters)


if __name__ == "__main__":
    train_model()
