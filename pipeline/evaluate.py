"""
Short description - This module contains code to evaluate a model

:author: Sarah Guido
"""
import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error


def generate_plot(y_true: np.ndarray, y_pred: np.ndarray, plot_filename: str) -> None:
    """
    Generate a plot of actual y values vs. predicted y values.

    :param y_true: Actual y values from original dataset.
    :param y_pred: Predictions for those y values using the trained model.
    :param plot_filename: Name of saved plot.
    """
    sns.set()
    assert len(y_true) == len(y_pred), "Lengths of actual and predicted do not match."
    assert plot_filename, "Filename cannot be None."

    vals = pd.DataFrame({"actual": y_true, "predicted": y_pred}).sort_values("actual")
    plt.figure(figsize=(12, 12))
    sns.lineplot(x="actual", y="actual", data=vals, color="green")
    sns.scatterplot(x="actual", y="predicted", data=vals)

    click.echo(f"Saving plot to {plot_filename}.")
    plt.savefig(plot_filename)


def generate_pred(model_file: str, x_test_file: str) -> np.ndarray:
    """
    Load the pretrained random forest regression model and generate predictions.

    :param model_file: Path to pretrained model.
    :param x_test_file: Path to X test data.
    :return: Predictions on the holdout test set.
    """
    click.echo(f"Loading pretrained model {model_file}.")

    rfr = load(model_file)
    assert rfr, "Model did not load."

    click.echo(f"Generating predictions with {rfr}.")

    X_test = pd.read_csv(x_test_file)
    y_pred = rfr.predict(X_test)

    return y_pred


@click.command()
@click.argument("x_test_file", type=click.Path(exists=True))
@click.argument("y_test_file", type=click.Path(exists=True))
@click.option("--model_file",
              type=click.Path(exists=True),
              required=True,
              help="Path to pretrained model file.")
@click.option("--metric",
              type=click.Choice(["rmse", "mae"]),
              default="rmse",
              help="Evaluation metric for random forest regression model. Default is rmse.")
@click.option("--plot",
              type=click.BOOL,
              default=False,
              help="Optionally generate and save a plot of actual vs. predicted y values. Default is false.")
@click.option("--plot_filename",
              type=click.Path(),
              default="actual_vs_predicted.png",
              help="Filename for saved plot. Default is actual_vs_predicted.png.")
def evaluate(x_test_file: str,
             y_test_file: str,
             model_file: str,
             metric: str = "rmse",
             plot: bool = None,
             plot_filename: str = None) -> None:
    """
    Evaluate a trained random forest regression model using either RMSE or MAE. Please pass in the paths to the test
    X and Y datasets. Optionally generate and save a plot.
    """
    y_true = pd.read_csv(y_test_file)["price"].values
    y_pred = generate_pred(model_file, x_test_file)

    assert len(y_true) == len(y_pred), "Actual y and predicted y must be the same length."
    assert isinstance(y_true, np.ndarray), "Actual y values should be an ndarray."
    assert isinstance(y_pred, np.ndarray), "Predicted y values should be an ndarray."

    click.echo(f"-------------Evaluating predictions using {metric}.-------------------")
    if metric == "rmse":
        click.echo(f"Root mean squared error: {np.sqrt(mean_squared_error(y_true, y_pred))}")
    else:
        click.echo(f"Mean absolute error: {mean_absolute_error(y_true, y_pred)}")

    if plot:
        click.echo("Generating plot of actual vs. predict values.")
        generate_plot(y_true, y_pred, plot_filename)


if __name__ == "__main__":
    evaluate()
