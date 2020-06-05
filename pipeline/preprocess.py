"""
Short description - This module contains code to process and prepare data for ML
modelling

:author: Sarah Guido
"""
import click
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def format_data(input_data: str,
                to_drop: str = None,
                log_price: bool = False) -> pd.DataFrame:
    """
    Format and split data for modeling.

    :param input_data: Path to file for preprocessing.
    :param to_drop: Optional. Additional columns to drop.
    :param log_price: Optional. Whether to take the log of the price column which will improve accuracy.
    """
    df = pd.read_csv(input_data)
    df = df.drop(["id", "date"], axis=1)

    if to_drop:
        to_drop = to_drop.split(",")

        assert set(to_drop) < set(df.columns), "Columns to drop must exist in dataframe."

        click.echo(f"Dropping additional columns: {to_drop}")
        df = df.drop(to_drop, axis=1)

    if log_price:
        click.echo("Taking the log of price.")
        assert 0 not in df["price"].values, "Cannot get the log of 0."
        df["price"] = np.log(df["price"])

    return df


def split_and_save(df: pd.DataFrame,
                   train_size: float,
                   X_train_file: str,
                   X_test_file: str,
                   y_train_file: str,
                   y_test_file: str) -> None:
    """
    Split the data into training and test sets, and then write out.

    :param df: DataFrame to be split.
    :param train_size: Size of the training set.
    :param X_train_file: Filename for X_train.
    :param X_test_file: Filename for X_test.
    :param y_train_file: Filename for y_train.
    :param y_test_file: Filename for y_test.
    """
    click.echo("Splitting data into training and test sets.")

    if train_size:
        assert 0.0 < train_size < 1.0, "Please pass a value between 0.0 and 1.0"

    X_train, X_test, y_train, y_test = train_test_split(df.drop("price", axis=1),
                                                        df["price"],
                                                        train_size=train_size)

    click.echo("Saving preprocessed split files.")

    X_train.to_csv(X_train_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)


@click.command()
@click.argument("input_data",
                type=click.Path(exists=True))
@click.option("--to_drop",
              type=str,
              default=None,
              help="Option to drop additional columns from dataset. Please pass in column names in the form of a "
                   "comma-separated list ex. bedrooms,bathrooms,floors without quotes.")
@click.option("--log_price",
              type=click.BOOL,
              default=False,
              help="Option to take the log of the price column. Please see the README for details on why this "
                   "is an option.")
@click.option("--train_size",
              type=float,
              default=None,
              help="Option to change the training size in train_test_split. For the sake of simplicity, please pass in "
                   "a float. The test size will be the complement of the training size.")
@click.option("--x_train_file", type=click.Path(), default="X_train", help="Path to save training features. Default "
                                                                           "filename is X_train.")
@click.option("--x_test_file", type=click.Path(), default="X_test", help="Path to save test features. Default "
                                                                         "filename is X_test.")
@click.option("--y_train_file", type=click.Path(), default="y_train", help="Path to save training predicted variables. "
                                                                           "Default filename is y_train.")
@click.option("--y_test_file", type=click.Path(), default="y_test", help="Path to save test predicted variables. Default"
                                                                         "filename is y_test.")
def preprocess(input_data: str,
               to_drop: str,
               log_price: bool,
               train_size: float,
               x_train_file: str,
               x_test_file: str,
               y_train_file: str,
               y_test_file: str):
    """
    Preprocess the input file for modeling. INPUT_DATA should be the path to a CSV file. Filenames for saved output
    are set up by default, but can be adjusted if you wish.
    """
    click.echo(f"Preprocessing file {input_data}")

    df = format_data(input_data, to_drop, log_price)

    split_and_save(df, train_size, x_train_file, x_test_file, y_train_file, y_test_file)


if __name__ == "__main__":
    preprocess()
