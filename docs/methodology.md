# Methodology

## Question

How do the number of neighbours (K) and input scaling change what K-Nearest Neighbours predicts, and how well?

## Data

| | Iris | Fruit |
| --- | --- | --- |
| Rows | 150 (50 per species) | 600 (150 per fruit) |
| Inputs | sepal length and width, petal length and width (cm) | weight (g), width and height (cm), colour score (0 to 1) |
| Classes | setosa, versicolor, virginica | apple, orange, lemon, mandarin |
| Source | scikit-learn's bundled UCI copy | generated, `knn_lab/data.py` |

The fruit rows are drawn from a normal distribution per fruit and measurement (`FRUIT_PROFILES`), with seed 42. Apples and oranges are given close averages so they overlap, as they do in practice.

## Steps

1. **Checks**: missing values, duplicates and ranges (`outputs/<dataset>/data_quality.csv`), class counts, pair plots and correlations.
2. **Split**: 70% training, 30% test, stratified so every class keeps its share. The test set is used once, at the end.
3. **Two models**: KNN with every input standardised (mean 0, spread 1) and KNN on the raw numbers. Standardising is a step inside the scikit-learn Pipeline, so it is fitted on training folds only.
4. **Tuning**: grid search with 5-fold stratified cross-validation over K (odd numbers 1 to 31, so two classes can't tie), vote weights (uniform or by distance) and distance (Euclidean or Manhattan). The best model is picked by cross-validation accuracy.
5. **Evaluation**: accuracy and macro precision, recall and F1 on the test set, confusion matrices and the list of wrong predictions.
6. **Which inputs matter**: permutation importance on the test set. Each input is shuffled 10 times and the drop in accuracy is recorded.
7. **K sweep**: with the other tuned settings fixed, K from 1 to 61, cross-validation and test accuracy for both models.
8. **Decision maps**: KNN trained on just two inputs, coloured by the predicted class across a grid, for K = 1, the tuned K and a very large K.
9. **In-depth analysis** (`pxlab.tabular.study`): 1,000-round bootstrap 95% intervals for the headline test scores, next to a baseline that ignores the inputs; McNemar's test of the best model against each other model; learning curves; and an error analysis that compares the inputs of right and wrong answers, lists the classes the model mixes up and shows how accuracy depends on the model's confidence. Each finding is one sentence in `outputs/<dataset>/key_findings.csv`.

## Limits

- Iris has only 45 test flowers, so one flower moves accuracy by 2.2 points. Differences that small are noise.
- The fruit data is generated from normal distributions. Real fruit has skewed weights and correlated measurements, so the exact accuracies don't carry over; the effect of scaling does.
- The decision maps use two inputs so they can be drawn. The real models use all four.
