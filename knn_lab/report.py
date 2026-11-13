"""
report.py - which saved tables and figures make up the results. The app's
Results tab and docs/results.md both read this list, so they always match.
"""


def results_sections(spec):
    """
    The results for one dataset, grouped into sections.

    Input:
        spec: the dataset SPEC.
    Returns:
        A list of (section name, items) for pxlab's show_results_tabs() and
        write_results_doc().
    """
    folder = spec["folder"]
    return [
        (
            "Models",
            [
                ("table", "Scaled against raw inputs", f"{folder}/model_comparison.csv"),
                ("figure", f"{folder}/confusion_matrices.png"),
            ],
        ),
        ("Mistakes", [("table", "Test examples the best model got wrong", f"{folder}/wrong_predictions.csv")]),
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
                ("figure", f"{folder}/k_sweep.png"),
                ("figure", f"{folder}/decision_maps.png"),
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
