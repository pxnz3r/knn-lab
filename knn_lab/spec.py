"""
spec.py - what each dataset looks like and which models to train.

The pxlab engine reads these dictionaries: which column to predict, which
columns are inputs, readable names, and the models with the settings that
grid search tries.
"""

from sklearn.neighbors import KNeighborsClassifier

# K values tried by grid search: odd numbers avoid ties between two classes.
K_VALUES = list(range(1, 32, 2))


def knn_models():
    """
    The two KNN variants compared on every dataset.

    "KNN" standardises every input first, so each one counts equally in the
    distance. "KNN without scaling" uses the raw numbers, so an input
    measured in big units (grams) drowns out one measured in small units
    (a 0 to 1 colour score). Comparing them shows why scaling matters.

    Returns:
        The SPEC["models"] dictionary.
    """
    grid = {"n_neighbors": K_VALUES, "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
    return {
        "KNN": {"estimator": KNeighborsClassifier(), "grid": grid, "scale": True, "role": "Main"},
        "KNN without scaling": {
            "estimator": KNeighborsClassifier(),
            "grid": grid,
            "scale": False,
            "role": "Comparison",
        },
    }


IRIS_SPEC = {
    "name": "Iris flowers",
    "folder": "iris",
    "task": "classification",
    "target": "species",
    "numeric": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "categorical": [],
    "labels": {
        "sepal_length": "Sepal length (cm)",
        "sepal_width": "Sepal width (cm)",
        "petal_length": "Petal length (cm)",
        "petal_width": "Petal width (cm)",
        "species": "Species",
    },
    "help": {
        "sepal_length": "The sepal is the green leaf-like part under the petals.",
        "sepal_width": "The sepal is the green leaf-like part under the petals.",
    },
    "test_size": 0.3,
    "scoring": "accuracy",
    "models": knn_models(),
    # The two inputs drawn on the decision-boundary maps.
    "map_inputs": ["petal_length", "petal_width"],
}

FRUIT_SPEC = {
    "name": "Fruit",
    "folder": "fruit",
    "task": "classification",
    "target": "fruit",
    "numeric": ["mass_g", "width_cm", "height_cm", "color_score"],
    "categorical": [],
    "labels": {
        "mass_g": "Weight (g)",
        "width_cm": "Width (cm)",
        "height_cm": "Height (cm)",
        "color_score": "Colour score",
        "fruit": "Fruit",
    },
    "help": {
        "color_score": "0 = deep red, 0.5 = orange, 1 = yellow-green, from a colour sensor.",
    },
    "test_size": 0.3,
    "scoring": "accuracy",
    "models": knn_models(),
    "map_inputs": ["mass_g", "color_score"],
}

SPECS = [IRIS_SPEC, FRUIT_SPEC]
