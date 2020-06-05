# ml_homework_problem

For this exercise, I built a little command line application called `ml_pipeline`. It's based on `Click`, a very nice library for building command line applications. 

## Directory and file overview

- `data`: original `home_data.csv` file.
- `example_output`: example output from running each module.
- `pipeline`: ml pipeline code.
    - `__init__.py`
    - `preprocess.py`
    - `train.py`
    - `evaluate.py`
- `exploratory.ipynb`: messing around with the data, model, and plots.
- `pytest.ini`: Pytest configuration file. In it I'm simply creating some test markers (explained in the Logging and Testing section below.)
- `requirements.txt`: necessary libraries are in here. Everything should get installed if you run `setup.py` as mentioned in the next section.
- `test_modules.py`: testing suite.

## Installing the modules

`cd` into the top-level directory of this project and create a virtual environment of your choice, using Python 3 (for example, something like `python3 -m venv ml_pipeline`). To install the three modules, run

`>> python setup.py install`

If you'd like to play around with the code more in-depth, feel free to run 

`>> python setup.py develop`

which will install everything in development mode. You won't have to reinstall anything while playing around.

## Module details

This section contains the thought process behind each module as well as documentation and examples for running each module.

### Preprocess

In this module, I do a minimal amount of preprocessing. Most of it is left up to the user - things like which columns to drop, where to store the output, and whether or not to take the `log` of the `price` column. 

Taking the `log` of the `price` column improves RMSE and MAE by quite a bit. However, I left this up to the user because I don't know how this output will be used. If, for example, the predicted housing prices are being delivered directly to some stakeholder, or is being communicated as "We're reasonably confident that this particular house will be worth $n", then it's more useful to deliver the actual predicted value. On the other hand, if the output is being used in some sort of machine learning system where minimizing the error of these metrics is more important than the raw value, we'd likely want to normalize `price`.

The other important thing that happens here is that the data is split into training and test sets into whatever split you'd like, utilizing `train_test_split`'s `train_size`. I decided to do that here because in the `train` step, we only need two files: `X_train` and `y_train` (or whatever you'd want to name them). It seemed reasonable to do it here and then only pass what's needed to the other two modules.

Here's the documentation for `preprocess`. After running `setup.py`, you should be able to type `preprocess --help` and see the following in your terminal:

```
>> preprocess --help
Usage: preprocess [OPTIONS] INPUT_DATA

  Preprocess the input file for modeling. INPUT_DATA should be the path to a
  CSV file. Filenames for saved output are set up by default, but can be
  adjusted if you wish.

Options:
  --to_drop TEXT       Option to drop additional columns from dataset.
                       Please pass in column names in the form of a comma-
                       separated list ex. bedrooms,bathrooms,floors without
                       quotes.

  --log_price BOOLEAN  Option to take the log of the price column. Please see
                       the README for details on why this is an option.

  --train_size FLOAT   Option to change the training size in train_test_split.
                       For the sake of simplicity, please pass in a float. The
                       test size will be the complement of the training size.

  --x_train_file PATH  Path to save training features. Default filename is
                       X_train.

  --x_test_file PATH   Path to save test features. Default filename is X_test.

  --y_train_file PATH  Path to save training predicted variables. Default
                       filename is y_train.

  --y_test_file PATH   Path to save test predicted variables. Defaultfilename
                       is y_test.

  --help               Show this message and exit.
```

An example of running this module is

`>> preprocess --to_drop sqft_living15,sqft_lot15 --log_price True --train_size 0.85 data/home_data.csv`

### Train

For training, we only need a few pieces of information: the `X_train` and `y_train` files generated during `preprocess`, a filename for the serialized model, and optionally, a series of hyperparameters to use in building the random forest regression model instead of the defaults.

Hyperparameters should be passed in like this:

`-hp n_estimators=50 -hp criteron=mae -hp n_jobs=4`

`joblib` is used to serialize the trained model. By default it will be named `rfr.joblib` but you can change that easily.

The other thing to note is that by default, verbosity is turned on while training the model. Random forests can be a little slow, depending on your combination of hyperparameters and if you've remembered to up the `n_jobs` parameter to get some parallelization going.

``` 
>> train --help
Usage: train [OPTIONS] X_TRAIN_PATH Y_TRAIN_PATH

  Train a Random Forest Regression model and save the model to disk. Please
  pass in the paths to the training X and Y datasets. Model is automatically
  set to verbose=2 in order to observe training progress. If you'd like to
  turn this behavior off, pass in -hp verbose=0 to the hyperparameter
  option.

Options:
  --model_filename PATH        Filename for saved model. Default is
                               rfr.joblib.

  -hp, --hyperparameters TEXT  Optional hyperparameters for random forest
                               regression model. For each parameter you want
                               to enter, please pass the -hp flag followed by
                               the hyperparameters in the format
                               parameter=value ex. -hp n_estimators=50 -hp
                               criterion=mae. Defaults to None.

  --help                       Show this message and exit.

```

Here's an example of running this module:

`>> train --model_filename example_output/rfr_file -hp n_estimators=200 -hp n_jobs=4 example_output/X_train example_output/y_train`

### Evaluate

Finally, here's the `evaluate` module. You can evaluate your predictions using either RMSE or MAE. Funnily enough, scikit-learn only has an MSE option, but it's easy enough to take the square root of that to get the RMSE.

You can output either RMSE or MAE. The module defaults to `rmse`.

You'll need the trained model serialized in the `train` step, as well as the `X_test` and `y_test` data from the `preprocess` step.

The other important thing here is generating the plot. By default, running this won't output a plot, but you can turn that on with `--plot True`, as well as decide where you want to save it beyond the default filename.

``` 
>> evaluate --help
Usage: evaluate [OPTIONS] X_TEST_FILE Y_TEST_FILE

  Evaluate a trained random forest regression model using either RMSE or
  MAE. Please pass in the paths to the test X and Y datasets. Optionally
  generate and save a plot.

Options:
  --model_file PATH     Path to pretrained model file.  [required]
  --metric [rmse|mae]   Evaluation metric for random forest regression model.
                        Default is rmse.

  --plot BOOLEAN        Optionally generate and save a plot of actual vs.
                        predicted y values. Default is false.

  --plot_filename PATH  Filename for saved plot. Default is
                        actual_vs_predicted.png.

  --help                Show this message and exit.
```

An example of running this module:

`>> evaluate --model_file example_output/rfr_file --plot True --plot_filename example_output/cool_plot example_output/X_test example_output/y_test`

## Logging and Testing

Throughout the modules, there are various messages that will pop up while running, informing you of where the program is and what's happening. 

In addition to the various `assert` statements throughout the code, there's a `test_modules.py` file to run unit tests on the functions in the modules. If you want to run tests on everything, make sure that `pytest` is installed in your virtualenv. Then you can run:

`>> pytest test_modules.py`

If you want to run tests for only a specific module, you can pass in either `preprocess`, `train`, or `evaluate` to the `-m` flag:

`>> pytest test_modules.py -m evaluate`

Note: a test I want to run on serializing a model is failing due to a `UnicodeDecodeError`, which doesn't happen when I run the `train` module outside of the test. I'm not sure why this is happening, but if I had more time, I'd certainly figure it out (unicode errors are my nightmare).

## If I had more time!

Some things I'd like to do if I had more time:

- I didn't spend a lot of time trying to build the best model possible. I would've liked to have tried some other ensemble models, maybe gradient-boosted trees or something, to compare performance.
- I don't have a lot of context for what this data set is or where it's from, and I didn't try to find out. I wasn't sure what some of the fields were (for example, `sqft_living` vs. `sqft_living15`). I like to know these things when building a model.
- It would've been cool to build this into an actual pipeline instead of three separate modules. `Click` seems like it would make this fairly straightforward with submodules and whanot, but I didn't spend time on figuring this out.
- I mentioned a weird `UnicodeDecodeError` that I couldn't figure out in the test suite.
- Making this generalizable to many more `sklearn` models would be really interesting and potentially useful. A generalizable `preprocess` module might be tricky, since that's dependent on the data and the model, but training and evaluating could possibly be better abstracted. Especially since models in scikit-learn follow the same `.fit()` pattern, and a lot of the evaluation metrics take in `(y_true, y_pred)` parameters.
