# Results

Produced by `python main.py`. Test sets are 30% of each dataset and were not used for tuning or model choice.

## Fruit

### Models

| Model | Role | CV accuracy | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| KNN | Main | 0.862 | 0.861 | 0.863 | 0.861 | 0.862 |
| KNN without scaling | Comparison | 0.705 | 0.694 | 0.706 | 0.694 | 0.695 |

### Permutation importance (drop in test accuracy when the input is shuffled)

| Input | KNN | KNN without scaling |
| --- | --- | --- |
| Colour score | 0.2178 | 0.0033 |
| Height (cm) | 0.1906 | 0.1622 |
| Weight (g) | 0.1322 | 0.3022 |
| Width (cm) | 0.1072 | 0.0617 |

### K sweep, scaled inputs

| n_neighbors | CV score | Test score |
| --- | --- | --- |
| 1 | 0.8214 | 0.8389 |
| 3 | 0.8024 | 0.8778 |
| 5 | 0.8238 | 0.8611 |
| 7 | 0.8286 | 0.8722 |
| 9 | 0.8429 | 0.8889 |
| 11 | 0.8357 | 0.8556 |
| 15 | 0.8452 | 0.8611 |
| 21 | 0.8429 | 0.8722 |
| 25 | 0.8571 | 0.8722 |
| 31 | 0.8595 | 0.85 |
| 41 | 0.8643 | 0.8389 |
| 51 | 0.8381 | 0.8389 |
| 61 | 0.8333 | 0.8278 |

### K sweep, raw inputs

| n_neighbors | CV score | Test score |
| --- | --- | --- |
| 1 | 0.6667 | 0.7 |
| 3 | 0.7048 | 0.6944 |
| 5 | 0.6929 | 0.7056 |
| 7 | 0.6667 | 0.6889 |
| 9 | 0.6476 | 0.6944 |
| 11 | 0.6595 | 0.7056 |
| 15 | 0.6571 | 0.7056 |
| 21 | 0.6548 | 0.7056 |
| 25 | 0.6667 | 0.7167 |
| 31 | 0.6571 | 0.7056 |
| 41 | 0.6738 | 0.7056 |
| 51 | 0.6738 | 0.6889 |
| 61 | 0.6619 | 0.6833 |

![Decision maps](../outputs/figures/fruit/decision_maps.png)

![Confusion matrices](../outputs/figures/fruit/confusion_matrices.png)

## Iris

### Models

| Model | Role | CV accuracy | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| KNN | Main | 0.971 | 0.933 | 0.944 | 0.933 | 0.933 |
| KNN without scaling | Comparison | 0.981 | 0.956 | 0.961 | 0.956 | 0.955 |

### Permutation importance (drop in test accuracy when the input is shuffled)

| Input | KNN | KNN without scaling |
| --- | --- | --- |
| Petal width (cm) | 0.2489 | 0.1578 |
| Petal length (cm) | 0.2111 | 0.5489 |
| Sepal length (cm) | 0.0222 | 0.0022 |
| Sepal width (cm) | 0.02 | 0.0 |

### K sweep, scaled inputs

| n_neighbors | CV score | Test score |
| --- | --- | --- |
| 1 | 0.9429 | 0.9333 |
| 3 | 0.9524 | 0.9333 |
| 5 | 0.9524 | 0.9333 |
| 7 | 0.9714 | 0.9333 |
| 9 | 0.9714 | 0.9333 |
| 11 | 0.9714 | 0.9333 |
| 15 | 0.9619 | 0.9333 |
| 21 | 0.9333 | 0.9111 |
| 25 | 0.9238 | 0.8889 |
| 31 | 0.9238 | 0.8889 |
| 41 | 0.9238 | 0.8889 |
| 51 | 0.9238 | 0.8889 |
| 61 | 0.9524 | 0.8889 |

### K sweep, raw inputs

| n_neighbors | CV score | Test score |
| --- | --- | --- |
| 1 | 0.9619 | 0.9333 |
| 3 | 0.9619 | 0.9556 |
| 5 | 0.9714 | 0.9778 |
| 7 | 0.981 | 0.9556 |
| 9 | 0.981 | 0.9556 |
| 11 | 0.981 | 0.9556 |
| 15 | 0.981 | 0.9778 |
| 21 | 0.9714 | 0.9556 |
| 25 | 0.9714 | 0.9556 |
| 31 | 0.9714 | 0.9333 |
| 41 | 0.9714 | 0.9111 |
| 51 | 0.9714 | 0.9111 |
| 61 | 0.9714 | 0.9111 |

![Decision maps](../outputs/figures/iris/decision_maps.png)

![Confusion matrices](../outputs/figures/iris/confusion_matrices.png)
