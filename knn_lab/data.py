"""
data.py - the two datasets.

Iris: 150 real flowers of three species, measured by Edgar Anderson and
used by R. A. Fisher in 1936. Shipped with scikit-learn (UCI Machine
Learning Repository, CC BY 4.0), so nothing is downloaded.

Fruit: generated here, because there is no clearly licensed public table
of fruit weights, sizes and colours. Each fruit type gets realistic
averages and spreads (see FRUIT_PROFILES) and the rows are drawn at random
with a fixed seed, so every run makes the same file. Apples and oranges
overlap on purpose, as they do in real life, so the choice of K matters.
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from pxlab.common import DATA_FOLDER, RANDOM_STATE

FRUIT_FILE = os.path.join(DATA_FOLDER, "fruit.csv")
FRUITS_PER_TYPE = 150

# Average and spread (standard deviation) of each measurement per fruit.
FRUIT_PROFILES = {
    "apple": {"mass_g": (165, 25), "width_cm": (7.5, 0.45), "height_cm": (7.6, 0.6), "color_score": (0.62, 0.12)},
    "orange": {"mass_g": (190, 35), "width_cm": (7.6, 0.5), "height_cm": (8.0, 0.7), "color_score": (0.55, 0.06)},
    "lemon": {"mass_g": (150, 40), "width_cm": (6.5, 0.45), "height_cm": (8.9, 0.9), "color_score": (0.82, 0.06)},
    "mandarin": {"mass_g": (85, 12), "width_cm": (6.0, 0.35), "height_cm": (4.6, 0.35), "color_score": (0.52, 0.05)},
}


def load_iris_data():
    """
    The Iris dataset as a table with readable column names.

    Returns:
        A DataFrame with four measurements and "species".
    """
    iris = load_iris(as_frame=True)
    data = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    data["species"] = [iris.target_names[code] for code in data.pop("target")]
    return data


def make_fruit_data():
    """
    Generate the fruit table from FRUIT_PROFILES and save it to data/fruit.csv.

    Returns:
        A DataFrame with mass_g, width_cm, height_cm, color_score and fruit.
    """
    generator = np.random.default_rng(RANDOM_STATE)
    parts = []
    for fruit, profile in FRUIT_PROFILES.items():
        part = pd.DataFrame(
            {column: generator.normal(mean, spread, FRUITS_PER_TYPE) for column, (mean, spread) in profile.items()}
        )
        part["fruit"] = fruit
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    data["mass_g"] = data["mass_g"].clip(lower=40).round(0)
    data[["width_cm", "height_cm"]] = data[["width_cm", "height_cm"]].clip(lower=2).round(1)
    data["color_score"] = data["color_score"].clip(0, 1).round(2)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    data.to_csv(FRUIT_FILE, index=False)
    return data


def load_dataset(spec):
    """
    The table for a SPEC.

    Input:
        spec: IRIS_SPEC or FRUIT_SPEC.
    Returns:
        A DataFrame.
    """
    if spec["folder"] == "iris":
        return load_iris_data()
    return make_fruit_data()
