"""
main.py - run the whole KNN study on both datasets with one command:

    python main.py

For each dataset (Iris flowers, then fruit) it:
    1. loads and checks the data, and draws its overview plots,
    2. keeps 30% of the rows back as a test set,
    3. tunes KNN (K, vote weights, distance) with 5-fold cross-validation,
       once with scaled inputs and once without,
    4. scores both on the test set and compares predictions with the real
       classes,
    5. measures which inputs matter (permutation importance),
    6. sweeps K from 1 to 60 and draws how the class boundaries change,
    7. saves the models for the app.
Figures and tables go to outputs/<dataset>/, models to models/<dataset>/.
"""

import os

from sklearn.base import clone

from knn_lab.data import load_dataset
from knn_lab.knn import plot_decision_maps
from knn_lab.spec import SPECS
from pxlab.common import MODELS_FOLDER, save_table, set_plot_style
from pxlab.tabular.data import (
    check_columns,
    input_summary,
    plot_correlations,
    plot_pairs,
    plot_target,
    quality_report,
    split_data,
)
from pxlab.tabular.evaluate import comparison_table, plot_confusion_matrices, test_scores, wrong_predictions
from pxlab.tabular.explain import (
    parameter_sweep,
    permutation_importance_table,
    plot_parameter_sweep,
    plot_permutation_importance,
)
from pxlab.tabular.train import best_model_name, save_models, train_models

SWEEP_K = [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 61]


def k_sweeps(spec, results, train_features, train_target, test_features, test_target):
    """
    Accuracy for every K in SWEEP_K, with and without scaling, keeping the
    other best settings fixed. Saves a table and two charts.

    Inputs:
        spec: the dataset SPEC.
        results: from train_models().
        train_features, train_target, test_features, test_target: the data.
    Returns:
        The table for the scaled model.
    """
    folder = spec["folder"]
    tables = {}
    for model_name in ["KNN", "KNN without scaling"]:
        model_spec = dict(spec["models"][model_name])
        best = results[model_name]["best_params"]
        model_spec["estimator"] = clone(model_spec["estimator"]).set_params(
            weights=best["weights"], metric=best["metric"]
        )
        table = parameter_sweep(
            spec, model_spec, "n_neighbors", SWEEP_K, train_features, train_target, test_features, test_target
        )
        tables[model_name] = table
        file_part = "k_sweep" if model_name == "KNN" else "k_sweep_no_scaling"
        save_table(table.round(4), f"{folder}/{file_part}.csv")
        plot_parameter_sweep(
            table,
            "n_neighbors",
            "Accuracy",
            f"{spec['name']}: accuracy against K ({model_name})",
            f"{folder}/{file_part}.png",
        )
    return tables["KNN"]


def run_dataset(spec):
    """
    The full study for one dataset.

    Input:
        spec: IRIS_SPEC or FRUIT_SPEC.
    Returns:
        Nothing.
    """
    folder = spec["folder"]
    print(f"\n=== {spec['name']} ===")
    data = load_dataset(spec)
    check_columns(data, spec)
    quality_report(data, spec, f"{folder}/data_quality.csv")
    plot_target(data, spec, f"{folder}/classes.png")
    plot_pairs(data, spec, f"{folder}/pairs.png")
    plot_correlations(data, spec, f"{folder}/correlations.png")

    train_features, test_features, train_target, test_target = split_data(data, spec)
    model_folder = os.path.join(MODELS_FOLDER, folder)
    os.makedirs(model_folder, exist_ok=True)
    train_features.assign(**{spec["target"]: train_target}).to_csv(os.path.join(model_folder, "train.csv"), index=False)
    test_features.assign(**{spec["target"]: test_target}).to_csv(os.path.join(model_folder, "test.csv"), index=False)

    print("Tuning KNN with 5-fold cross-validation:")
    results = train_models(spec, train_features, train_target)
    best = best_model_name(results)

    table = comparison_table(results, test_features, test_target, spec, f"{folder}/model_comparison.csv")
    print(table.to_string(index=False))
    plot_confusion_matrices(results, test_features, test_target, f"{folder}/confusion_matrices.png")
    wrong = wrong_predictions(results[best]["pipeline"], test_features, test_target, spec)
    save_table(wrong, f"{folder}/wrong_predictions.csv")
    print(f"  {best} got {len(wrong)} of {len(test_target)} test examples wrong")

    importance = permutation_importance_table(
        results, test_features, test_target, spec, filename=f"{folder}/permutation_importance.csv"
    )
    plot_permutation_importance(importance, filename=f"{folder}/permutation_importance.png")

    sweep = k_sweeps(spec, results, train_features, train_target, test_features, test_target)
    best_k = int(results["KNN"]["best_params"]["n_neighbors"])
    largest_k = int(len(train_features) * 0.4)
    if largest_k % 2 == 0:
        largest_k += 1  # odd, so two classes can't tie
    plot_decision_maps(train_features, train_target, spec, [1, best_k, largest_k], f"{folder}/decision_maps.png")

    model_info = {
        "dataset": spec["name"],
        "classes": sorted(str(c) for c in data[spec["target"]].unique()),
        "best_model": best,
        "test_scores": {
            name: test_scores(r["pipeline"], test_features, test_target, spec) for name, r in results.items()
        },
        "cv_scores": {name: r["cv_score"] for name, r in results.items()},
        "k_sweep_best_test_k": int(sweep.loc[sweep["Test score"].idxmax(), "n_neighbors"]),
    }
    model_info.update(input_summary(train_features, spec))
    save_models(results, model_info, folder=model_folder)


def main():
    """
    Run the study on every dataset.

    Returns:
        Nothing.
    """
    set_plot_style()
    for spec in SPECS:
        run_dataset(spec)
    print("\nDone. Figures and tables are in outputs/, models in models/. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
