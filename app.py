"""
app.py - the KNN lab demo. Run it with:

    streamlit run app.py

Pick a dataset, then:
    Overview      what the project is and how KNN works,
    Classify      enter measurements, see the predicted class and the
                  neighbours that voted for it,
    Effect of K   move K and watch the test accuracy change,
    Results       the saved tables and figures from main.py.
It needs the saved models, so run `python main.py` once first.
"""

import os

import streamlit as st

from knn_lab.knn import accuracy_for_k, nearest_neighbours, split_frames
from knn_lab.spec import SPECS
from pxlab.app import (
    class_probabilities,
    input_form,
    probability_bar,
    setup_page,
    show_figure,
    show_markdown_file,
    show_results_tabs,
)
from pxlab.common import MODELS_FOLDER
from pxlab.tabular.data import label_for
from pxlab.tabular.train import load_models, models_are_saved

SPECS_BY_NAME = {spec["name"]: spec for spec in SPECS}


@st.cache_resource
def load_dataset_models(folder):
    """
    Load the saved models and the training and test sets for one dataset
    (cached, so this happens once per dataset).

    Input:
        folder: the dataset folder name inside models/.
    Returns:
        (model_info, models, train_features, train_target, test_features, test_target)
    """
    model_folder = os.path.join(MODELS_FOLDER, folder)
    model_info, models = load_models(model_folder)
    spec = next(spec for spec in SPECS if spec["folder"] == folder)
    frames = split_frames(os.path.join(model_folder, "train.csv"), os.path.join(model_folder, "test.csv"), spec)
    return (model_info, models, *frames)


@st.cache_data
def k_accuracy(folder, k, scale):
    """
    Test accuracy of KNN with one K (cached per dataset, K and scaling).

    Inputs:
        folder: the dataset folder name.
        k: the number of neighbours.
        scale: whether the inputs are standardised.
    Returns:
        The test accuracy.
    """
    _, _, train_features, train_target, test_features, test_target = load_dataset_models(folder)
    return accuracy_for_k(train_features, train_target, test_features, test_target, k, scale)


def typical_example(train_features, train_target, class_name):
    """
    The median measurements of one class, used as a starting point.

    Inputs:
        train_features, train_target: the training data.
        class_name: the class.
    Returns:
        A dictionary column -> value.
    """
    return train_features[train_target == class_name].median().round(2).to_dict()


def show_classify_page(spec, model_info, models, train_features, train_target):
    """
    The Classify tab: inputs on the left, the prediction and the
    neighbours that voted on the right.

    Inputs:
        spec: the dataset SPEC.
        model_info, models: from load_models().
        train_features, train_target: the training data, for the neighbours.
    Returns:
        Nothing.
    """
    left, right = st.columns([3, 2], gap="large")
    with left:
        picker, model_box = st.columns(2)
        start_class = picker.selectbox(
            "Start from a typical",
            model_info["classes"],
            help="Fills the inputs with the median measurements of that class.",
        )
        model_name = model_box.selectbox(
            "Model",
            list(models),
            format_func=lambda name: name + (" · best" if name == model_info["best_model"] else ""),
        )
        start_values = typical_example(train_features, train_target, start_class)
        example = input_form(spec, model_info, start_values, key_prefix=f"{spec['folder']}-{start_class}-")

    pipeline = models[model_name]
    probabilities = class_probabilities(pipeline, example)
    predicted = probabilities.index[0]
    k = model_info["best_params"][model_name]["n_neighbors"]
    with right:
        st.subheader(f"Predicted: {predicted}", anchor=False)
        st.caption(f"{model_name}, K = {k}, {model_info['best_params'][model_name]['weights']} votes")
        for class_name, probability in probabilities.items():
            st.markdown(f"**{class_name}** · {probability:.0%}")
            probability_bar(probability)

    st.subheader(f"The {k} nearest training examples", anchor=False)
    st.caption("These are the neighbours that voted. Distance is measured after the model's own scaling.")
    neighbours = nearest_neighbours(pipeline, example, train_features, train_target)
    st.dataframe(
        neighbours.rename(columns=lambda column: label_for(spec, column)),
        hide_index=True,
        width="stretch",
    )


def show_k_page(spec, model_info, train_features):
    """
    The Effect of K tab: a K slider with live test accuracy, then the saved
    sweep and decision-map figures.

    Inputs:
        spec: the dataset SPEC.
        model_info: from load_models().
        train_features: the training data (its size limits K).
    Returns:
        Nothing.
    """
    largest = min(99, len(train_features) - 1)
    if largest % 2 == 0:
        largest -= 1
    best_k = model_info["best_params"]["KNN"]["n_neighbors"]
    k = st.slider("K, the number of neighbours that vote", 1, largest, best_k, step=2)
    scaled, unscaled = st.columns(2)
    scaled.metric("Test accuracy, scaled inputs", f"{k_accuracy(spec['folder'], k, True):.1%}")
    unscaled.metric("Test accuracy, raw inputs", f"{k_accuracy(spec['folder'], k, False):.1%}")
    st.caption(
        f"Cross-validation picked K = {best_k}. Very small K copies the noise of single examples; "
        "very large K outvotes the smaller classes."
    )
    show_figure(f"{spec['folder']}/k_sweep.png")
    show_figure(f"{spec['folder']}/decision_maps.png")


def show_results_page(spec):
    """
    The Results tab: everything main.py saved for this dataset.

    Input:
        spec: the dataset SPEC.
    Returns:
        Nothing.
    """
    folder = spec["folder"]
    show_results_tabs(
        [
            (
                "Models",
                [
                    ("table", "Scaled against raw inputs", f"{folder}/model_comparison.csv"),
                    ("figure", f"{folder}/confusion_matrices.png"),
                ],
            ),
            (
                "Mistakes",
                [("table", "Test examples the best model got wrong", f"{folder}/wrong_predictions.csv")],
            ),
            (
                "Which inputs matter",
                [
                    ("figure", f"{folder}/permutation_importance.png"),
                    ("table", "Permutation importance", f"{folder}/permutation_importance.csv"),
                ],
            ),
            (
                "K sweep",
                [
                    ("table", "Scaled inputs", f"{folder}/k_sweep.csv"),
                    ("table", "Raw inputs", f"{folder}/k_sweep_no_scaling.csv"),
                    ("figure", f"{folder}/k_sweep_no_scaling.png"),
                ],
            ),
            (
                "Data",
                [
                    ("table", "Data quality", f"{folder}/data_quality.csv"),
                    ("figure", f"{folder}/classes.png"),
                    ("figure", f"{folder}/pairs.png"),
                    ("figure", f"{folder}/correlations.png"),
                ],
            ),
        ]
    )


def main():
    """
    Build the page.

    Returns:
        Nothing.
    """
    setup_page("KNN lab")
    dataset_name = st.radio("Dataset", list(SPECS_BY_NAME), horizontal=True)
    spec = SPECS_BY_NAME[dataset_name]
    if not models_are_saved(os.path.join(MODELS_FOLDER, spec["folder"])):
        st.error("No saved models yet. Run `python main.py` first.")
        st.stop()
    model_info, models, train_features, train_target, _, _ = load_dataset_models(spec["folder"])

    overview, classify, effect_of_k, results = st.tabs(["Overview", "Classify", "Effect of K", "Results"])
    with overview:
        show_markdown_file("app_overview.md")
    with classify:
        show_classify_page(spec, model_info, models, train_features, train_target)
    with effect_of_k:
        show_k_page(spec, model_info, train_features)
    with results:
        show_results_page(spec)


main()
