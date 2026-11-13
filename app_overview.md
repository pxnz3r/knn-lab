### What this is

A study of **K-Nearest Neighbours (KNN)**, one of the simplest classifiers there is. It has two datasets:

- **Iris flowers**: 150 real flowers of three species, each measured four ways (sepal and petal length and width).
- **Fruit**: 600 apples, oranges, lemons and mandarins described by weight, width, height and a colour score. These rows are generated, with realistic averages and spreads per fruit. Apples and oranges overlap on purpose.

### How KNN decides

KNN doesn't learn rules. It keeps every training example. To classify a new one it:

1. measures the distance from the new example to every training example,
2. takes the **K** closest ones,
3. lets them vote; the class with the most votes wins. With *distance* votes, closer neighbours count more.

Two choices decide how well that works:

- **K.** With K = 1 a single odd example can flip the answer (overfitting). With K very large the vote is spread so thin that small classes lose (underfitting). Cross-validation picks K on the training data only.
- **Scaling.** Distance adds up differences in every input. Weight in grams runs into the hundreds while the colour score runs from 0 to 1, so without scaling weight drowns out colour. Standardising every input first makes them count equally.

### What to try

- **Classify**: start from a typical flower or fruit, change a measurement and watch the prediction and the neighbours that voted for it.
- **Effect of K**: drag K and compare scaled and raw inputs on the test set.
- **Results**: the model comparison, mistakes, which inputs matter, and the data checks.

### What was found

- On **fruit**, scaling is the difference between 86% and 69% test accuracy. Without it, the model barely looks at colour (permutation importance 0.003), which is the input that separates lemons from the rest.
- On **Iris**, all four measurements are already in centimetres on similar ranges, so scaling doesn't help (93% scaled, 96% raw on 45 test flowers, a difference of one flower).
- Petal measurements carry almost all the information on Iris; sepals add little.

Training code and every figure: [github.com/pxnz3r/knn-lab](https://github.com/pxnz3r/knn-lab).
