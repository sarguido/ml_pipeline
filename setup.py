#!/usr/bin/env python

from setuptools import setup

setup(
      name="ml_pipeline",
      version="0.1",
      packages=["pipeline"],
      package_data={},
      install_requires=["click", "pandas", "sklearn", "seaborn", "numpy"],
      entry_points={
        "console_scripts": ["preprocess = pipeline.preprocess:preprocess",
                            "train = pipeline.train:train_model",
                            "evaluate = pipeline.evaluate:evaluate"]
      }
)
