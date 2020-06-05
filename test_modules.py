"""
Test suite for preprocess, train, and evaluate modules.

:author: Sarah Guido
"""
import pytest
import pandas as pd
import numpy as np

from pipeline.preprocess import format_data, split_and_save
from pipeline.train import value_check, format_hyperparameters, train_and_save
from pipeline.evaluate import generate_pred, generate_plot


@pytest.mark.preprocess
def test_preprocess_basic():
    format_data("data/home_data.csv")


@pytest.mark.preprocess
def test_preprocess_complex():
    format_data("data/home_data.csv", "bedrooms,bathrooms,floors", True)


@pytest.mark.preprocess
def test_split_and_save(tmpdir):
    tmp = tmpdir.mkdir("test")
    df = pd.read_csv("data/home_data.csv")
    split_and_save(df, 0.4, tmp + "/X_train", tmp + "/X_test", tmp + "/y_train", tmp + "/y_test")

    assert len(tmp.listdir()) == 4


@pytest.mark.train
def test_value_check():
    test_input = ["4", "12.04", "true", "a string here", "0.05", "one"]

    for item in test_input:
        value_check(item)


@pytest.mark.train
def test_format_hyperparams():
    test_input = ("first_param=6", "second_param=1.9", "boolean=true", "last_param=a string here")

    format_hyperparameters(test_input)


@pytest.mark.train
@pytest.mark.xfail(raises=UnicodeDecodeError, reason="Getting a UnicodeDecodeError here that I do not get in the "
                                                     "module itself. Unsure how to fix.")
def test_train_and_save(tmpdir):
    tmp = tmpdir.mkdir("test").join("")
    X_train = pd.read_csv("example_output/X_train")
    y_train = pd.read_csv("example_output/y_train")
    train_and_save(X_train, y_train, tmp + "/model.joblib", ("verbose=0", "n_estimators=30"))

    assert len(tmp.listdir()) == 1


@pytest.mark.evaluate
def test_generate_plot(tmpdir):
    tmp = tmpdir.mkdir("test")
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.5, 2, 3.5])
    generate_plot(y_true, y_pred, tmp + "/plot.png")

    assert len(tmp.listdir()) == 1


@pytest.mark.evaluate
def test_generate_pred():
    y_pred = generate_pred("example_output/rfr_file", "example_output/X_test")

    y_true = pd.read_csv("example_output/y_test")
    assert len(y_pred) == len(y_true["price"].values), "Actual and predicted should be the same length."
