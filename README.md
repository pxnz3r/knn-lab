<div align="center">

<h1>KNN lab</h1>

<p>K-Nearest Neighbours on Iris flowers and fruit: how the number of neighbours (K) and input scaling change the predictions, with an interactive demo.</p>

<p>
<a href="https://github.com/pxnz3r/knn-lab/actions/workflows/ci.yml"><img src="https://github.com/pxnz3r/knn-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.14-blue" alt="Python 3.14">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

<p><a href="docs/results.md">Results</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/deploy.md">Deploy the demo</a></p>

<img src="outputs/figures/fruit/decision_maps.png" width="720" alt="Decision regions for K = 1, the tuned K and a very large K">

</div>

## How it works

1. **Load and check** Iris (150 flowers) and a generated fruit table (600 fruit).
2. **Tune K**, vote weights and distance with 5-fold cross-validation, with and without scaled inputs.
3. **Score once** on a 30% test split and explain the predictions with permutation importance and decision maps.
4. **Test how sure the scores are** with bootstrap intervals, a paired test against every rival, learning curves and a baseline that ignores the inputs.
5. **Study the mistakes**: which classes get confused and how accuracy rises with confidence, then **try it** in the Streamlit demo.

## Results

Test set: 30% of each dataset, never used for tuning. K, vote weights and distance were tuned with 5-fold cross-validation on the training set.

| Dataset | Model | K | CV accuracy | Test accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| Fruit (600) | KNN, scaled inputs | 29 | 0.862 | **0.861** | 0.862 |
| Fruit (600) | KNN, raw inputs | 3 | 0.705 | 0.694 | 0.695 |
| Iris (150) | KNN, scaled inputs | 7 | 0.971 | 0.933 | 0.933 |
| Iris (150) | KNN, raw inputs | 7 | 0.981 | **0.956** | 0.955 |

- **Scaling matters when units differ.** Fruit weight runs into the hundreds of grams and the colour score from 0 to 1. Without scaling, KNN ignores colour (permutation importance 0.003 against 0.218 scaled), and colour is what tells lemons apart.
- **It doesn't when units match.** Iris measurements are all centimetres on similar ranges; the gap between the two models is one test flower.
- **K trades noise for detail.** K = 1 follows every stray example; on fruit, accuracy climbs to K of about 25 to 41 and then falls as the smaller groups get outvoted. See `outputs/figures/*/k_sweep.png`.

Full tables: [docs/results.md](docs/results.md).

## How sure the results are

**Iris is too small to separate the two models.** Raw inputs score 0.956 with a 95% bootstrap interval of 0.889 to 1.000; against scaled inputs McNemar's test gives p = 1.0.

**On fruit, scaling is a real win.** Scaled KNN scores 0.861 (0.806 to 0.911) and beats raw inputs on McNemar's test (p < 0.001). Without scaling, KNN memorises the training fruit: training accuracy 1.000 against 0.705 on validation.

**Oranges are the hard case.** 24% of oranges are called apples. When scaled KNN is at least 90% sure, it is always right, on 44% of the test fruit.

Every table and figure behind these is in [docs/results.md](docs/results.md).

## Run it

Needs Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python main.py
uv run streamlit run app.py
```

`main.py` writes figures and tables to `outputs/` and models to `models/` (both are committed, so the demo works without running it). Iris ships with scikit-learn; the fruit table is generated with a fixed seed, so nothing is downloaded.

## The demo

- **Overview**: what KNN is and what was found, for someone arriving from a link.
- **Classify**: start from a typical flower or fruit, change the measurements, and see the prediction, the class probabilities and the K neighbours that voted.
- **Effect of K**: drag K and watch test accuracy with scaled and raw inputs side by side.
- **Results**: model comparison, mistakes, permutation importance, K sweep and data checks.

## Layout

```
knn_lab/      spec.py (datasets and models), data.py, knn.py (decision maps, neighbours)
pxlab/        shared engine (vendored): data checks, pipelines, tuning, evaluation, app helpers
main.py       the full study
app.py        the Streamlit demo
docs/         methodology, results, deploy
```

## Data

- **Iris**: Fisher (1936) via scikit-learn's bundled copy of the [UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris), CC BY 4.0.
- **Fruit**: generated in `knn_lab/data.py` from per-fruit averages and spreads. It is a teaching dataset, not real measurements.

## License

MIT
