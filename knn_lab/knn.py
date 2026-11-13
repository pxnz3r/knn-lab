"""
knn.py - pictures and helpers that show how K-Nearest Neighbours decides.

KNN keeps every training example. To classify a new one it finds the K
training examples closest to it (by distance between their measurements)
and lets them vote. Small K follows every single neighbour, so one odd
example can flip a prediction (overfitting). Large K averages over many
neighbours, so the boundaries between classes get smooth, and with K too
large the small classes get outvoted (underfitting).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pxlab.common import save_figure
from pxlab.tabular.data import label_for


def plot_decision_maps(train_features, train_target, spec, k_values, filename):
    """
    Colour a map of two inputs by the class KNN would predict there, for
    several values of K, with the training examples on top.

    Only two inputs are used so the map can be drawn on paper; the real
    models use all four.

    Inputs:
        train_features, train_target: the training data.
        spec: the dataset SPEC (its "map_inputs" names the two inputs).
        k_values: the K values to draw, one map each.
        filename: the image file name.
    Returns:
        The saved path.
    """
    x_name, y_name = spec["map_inputs"]
    points = train_features[[x_name, y_name]].to_numpy()
    classes = sorted(train_target.unique())
    codes = train_target.map({name: code for code, name in enumerate(classes)}).to_numpy()
    x_grid, y_grid = np.meshgrid(
        np.linspace(points[:, 0].min() - 0.5 * points[:, 0].std(), points[:, 0].max() + 0.5 * points[:, 0].std(), 300),
        np.linspace(points[:, 1].min() - 0.5 * points[:, 1].std(), points[:, 1].max() + 0.5 * points[:, 1].std(), 300),
    )
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    # One colour per class, used for both the regions and the dots.
    palette = [plt.get_cmap("Set2")(code) for code in range(len(classes))]
    region_colours = ListedColormap(palette)

    figure, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 4.6), squeeze=False)
    for axis, k in zip(axes.flat, k_values):
        model = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        model.fit(points, codes)
        predicted = model.predict(grid_points).reshape(x_grid.shape)
        axis.contourf(
            x_grid, y_grid, predicted, levels=np.arange(len(classes) + 1) - 0.5, cmap=region_colours, alpha=0.35
        )
        for code, name in enumerate(classes):
            chosen = codes == code
            axis.scatter(
                points[chosen, 0],
                points[chosen, 1],
                s=14,
                color=palette[code],
                edgecolor="black",
                linewidth=0.3,
                label=name,
            )
        axis.set_title(f"K = {k}")
        axis.set_xlabel(label_for(spec, x_name))
        axis.set_ylabel(label_for(spec, y_name))
    axes.flat[0].legend(loc="best", fontsize=8)
    figure.suptitle(f"{spec['name']}: what KNN predicts across two inputs as K grows")
    figure.tight_layout()
    return save_figure(figure, filename)


def nearest_neighbours(pipeline, example, train_features, train_target):
    """
    The training examples a fitted KNN pipeline uses to classify one example.

    Inputs:
        pipeline: a fitted KNN pipeline (steps "preprocess" and "model").
        example: a one-row DataFrame.
        train_features, train_target: the data the pipeline was trained on,
            in the same order.
    Returns:
        A DataFrame of the K neighbours, nearest first, with their class
        and distance.
    """
    model = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocess"].transform(example)
    distances, indices = model.kneighbors(transformed)
    neighbours = train_features.iloc[indices[0]].copy()
    neighbours.insert(0, "Class", train_target.iloc[indices[0]].to_numpy())
    neighbours["Distance"] = distances[0].round(3)
    return neighbours.reset_index(drop=True)


def accuracy_for_k(train_features, train_target, test_features, test_target, k, scale=True):
    """
    Train KNN with one K and score it on the test set (used live in the app).

    Inputs:
        train_features, train_target, test_features, test_target: the data.
        k: the number of neighbours.
        scale: whether to standardise the inputs first.
    Returns:
        The test accuracy.
    """
    steps = [("scale", StandardScaler())] if scale else []
    model = Pipeline(steps + [("knn", KNeighborsClassifier(n_neighbors=k))])
    model.fit(train_features, train_target)
    return float((model.predict(test_features) == test_target).mean())


def split_frames(train_csv, test_csv, spec):
    """
    Read the saved training and test sets back into features and targets.

    Inputs:
        train_csv, test_csv: paths of the CSV files main.py saved.
        spec: the dataset SPEC.
    Returns:
        (train_features, train_target, test_features, test_target)
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    inputs = list(spec["numeric"]) + list(spec["categorical"])
    return train[inputs], train[spec["target"]], test[inputs], test[spec["target"]]
